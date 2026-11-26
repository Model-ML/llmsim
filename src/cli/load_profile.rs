// Shared opt-in load settings for the three HTTP text-generation endpoints.

use super::{AppState, Config, ConfigError};
use crate::errors::{ErrorInjector, SimulatedError};
use crate::latency::{LatencyProfile, StreamPacer};
use crate::scenario::{self, ConversationMessage, MessageRole, Resolution, ValueRange};
use axum::http::HeaderMap;
use axum::response::Response;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenEstimate {
    #[default]
    Tiktoken,
    Bytes,
}

impl TokenEstimate {
    pub(super) fn count(self, text: &str) -> usize {
        match self {
            Self::Tiktoken => crate::count_tokens_default(text)
                .unwrap_or_else(|_| text.split_whitespace().count()),
            Self::Bytes => text.len() / 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProfile {
    pub ttft_ms: ValueRange,
    pub tokens_per_second: f64,
    pub output_tokens: ValueRange,
}

impl ModelProfile {
    pub(super) fn validate(&self, pattern: &str) -> Result<(), ConfigError> {
        let context =
            |error: String| ConfigError::Validation(format!("model_profiles.{pattern}: {error}"));
        self.ttft_ms.validate("ttft_ms").map_err(context)?;
        self.output_tokens
            .validate("output_tokens")
            .map_err(context)?;
        if pattern.is_empty() || !(0.001..=1_000_000.0).contains(&self.tokens_per_second) {
            return Err(context(
                "expected a nonempty glob and tokens_per_second between 0.001 and 1000000".into(),
            ));
        }
        Ok(())
    }
}

impl Config {
    pub(super) fn model_profile(&self, model: &str) -> Option<&ModelProfile> {
        self.model_profiles
            .iter()
            .filter(|(pattern, _)| glob_matches(pattern, model))
            .max_by_key(|(pattern, _)| {
                let wildcards = pattern.bytes().filter(|b| matches!(b, b'*' | b'?')).count();
                (
                    wildcards == 0,
                    pattern.len() - wildcards,
                    std::cmp::Reverse(wildcards),
                    std::cmp::Reverse(*pattern),
                )
            })
            .map(|(_, profile)| profile)
    }

    pub(super) fn generation_latency(&self, model: &str) -> LatencyProfile {
        let Some(profile) = self.model_profile(model) else {
            return self.latency_for_model(model);
        };
        LatencyProfile::new(
            profile.ttft_ms.sample_random(),
            0,
            (1000.0 / profile.tokens_per_second).round() as u64,
            0,
        )
        .with_prefill_us_per_token(self.latency.prefill_us_per_token.unwrap_or(0))
    }

    pub(super) fn generate_text(
        &self,
        request: &crate::openai::ChatCompletionRequest,
        use_profile: bool,
    ) -> String {
        let name = &self.response.generator;
        let profile = use_profile
            .then(|| self.model_profile(&request.model))
            .flatten();
        if let Some(profile) = profile {
            // Fixed and echo modes have explicit content. Length-based generators
            // oversample their word target so it can be capped in model tokens.
            let generator = name.to_ascii_lowercase();
            if generator != "echo" && !generator.starts_with("fixed:") {
                let cap = request
                    .max_completion_tokens
                    .or(request.max_tokens)
                    .unwrap_or(32_768) as u64;
                let tokens = profile
                    .output_tokens
                    .sample_random()
                    .clamp(1, 32_768)
                    .min(cap) as usize;
                let text = crate::create_generator(name, tokens * 2).generate(request);
                return crate::tokens::truncate_output_tokens(text, tokens);
            }
        }
        crate::create_generator(name, self.response.target_tokens).generate(request)
    }

    pub(super) fn response_delay(
        &self,
        model: &str,
        latency: &LatencyProfile,
        output_tokens: usize,
    ) -> Duration {
        let ttft = latency.sample_ttft();
        self.model_profile(model).map_or(ttft, |profile| {
            ttft + Duration::from_secs_f64(output_tokens as f64 / profile.tokens_per_second)
        })
    }
}

impl AppState {
    pub(super) fn stream_pacer(&self, model: &str, tokens: usize) -> Option<StreamPacer> {
        let profile = self.config.model_profile(model)?;
        let stats = self.load_stats.clone();
        let model = model.to_owned();
        Some(StreamPacer::new(
            tokens as u64,
            profile.tokens_per_second,
            move |tokens, duration| {
                stats.record_rate(&model, tokens, duration);
            },
        ))
    }

    pub(super) fn scenario_pacer(
        &self,
        model: &str,
        tokens: usize,
        resolution: &Resolution,
    ) -> Option<StreamPacer> {
        let timing = &resolution.timing;
        if timing.inter_token.is_some() || timing.speed != 1.0 || timing.stall.is_some() {
            return None;
        }
        self.stream_pacer(model, tokens)
    }
}

fn glob_matches(pattern: &str, model: &str) -> bool {
    let mut pattern = pattern.chars().peekable();
    let mut model = model.chars().peekable();
    let mut retry = None;
    while let Some(&ch) = model.peek() {
        match pattern.peek() {
            Some('*') => {
                pattern.next();
                retry = Some((pattern.clone(), model.clone()));
            }
            Some(&p) if p == '?' || p == ch => {
                pattern.next();
                model.next();
            }
            _ => {
                let Some((saved_pattern, remaining)) = retry.as_mut() else {
                    return false;
                };
                if remaining.next().is_none() {
                    return false;
                }
                pattern = saved_pattern.clone();
                model = remaining.clone();
            }
        }
    }
    pattern.all(|p| p == '*')
}

/// Faults use the SDK attempt header when a run seed is configured. Inline
/// playback always uses the header, so it never uses an attempt counter.
#[derive(Clone, Copy)]
pub(super) struct Attempt {
    pub(super) index: u32,
    run_seed: Option<i64>,
    fault_key: Option<u64>,
}

impl Attempt {
    pub(super) fn new(
        config: &Config,
        headers: &HeaderMap,
        messages: &[ConversationMessage],
    ) -> Self {
        let index = headers
            .get("x-stainless-retry-count")
            .and_then(|value| value.to_str().ok()?.parse().ok())
            .unwrap_or(0);
        let fault_key = config.errors.seed.map(|run_seed| {
            let user = messages.iter().position(|m| m.role == MessageRole::User);
            let text = user.map_or("", |i| messages[i].text.as_str());
            let session = scenario::turn_shape_payload(text).map_or_else(
                || crate::schema_fake::seed_from(text.as_bytes()),
                |(payload, seed)| scenario::turn_shape_session(payload, seed),
            );
            let step = scenario::tool_rounds(&messages[user.map_or(0, |i| i + 1)..]);
            scenario::fault_seed(run_seed, session, step, index)
        });
        Self {
            index,
            run_seed: config.errors.seed,
            fault_key,
        }
    }

    pub(super) fn for_step(self, state: &AppState, resolution: &Resolution, inline: bool) -> u32 {
        if inline || self.run_seed.is_some() {
            self.index
        } else {
            state.attempts.next_attempt(resolution.fingerprint)
        }
    }

    pub(super) fn outcome(self, resolution: &Resolution, index: u32) -> scenario::AttemptOutcome {
        resolution.outcome_with_seed(index, self.run_seed)
    }

    pub(super) fn injected_error(self, config: &Config) -> Option<SimulatedError> {
        let injector = ErrorInjector::new(config.error_config());
        if !injector.is_enabled() {
            return None;
        }
        match self.fault_key {
            Some(key) => injector.inject_seeded(key),
            None => injector.maybe_inject(),
        }
    }

    pub(super) fn overload(
        self,
        state: &AppState,
        provider: super::limits::Provider,
    ) -> Option<Response> {
        let rate = state.config.errors.overload_burst_rate;
        if rate <= 0.0 {
            return None;
        }
        let key = self.fault_key?;
        let draw = scenario::fault_seed(0, key, 0, 0) >> 11;
        if (draw as f64) / ((1u64 << 53) as f64) >= rate {
            return None;
        }
        let (status, kind) = match provider {
            super::limits::Provider::Anthropic => (529, "overloaded_error"),
            _ => (503, "service_unavailable"),
        };
        state.stats.record_error(status);
        Some(super::limits::with_retry_after(
            provider.error(status, kind, "Overloaded"),
            Duration::from_millis(state.config.errors.overload_burst_ms),
        ))
    }
}

#[derive(Default)]
pub(super) struct LoadStats {
    pub(super) shape_tool_substitutions: AtomicU64,
    rates: Mutex<HashMap<String, RateSample>>,
}

#[derive(Default)]
struct RateSample {
    tokens: u64,
    duration: Duration,
    streams: u64,
}

#[derive(Serialize)]
pub(super) struct ModelRate {
    configured_tokens_per_second: f64,
    achieved_tokens_per_second: Option<f64>,
    completed_streams: u64,
    output_tokens: u64,
}

impl LoadStats {
    pub(super) fn record_rate(&self, model: &str, tokens: u64, duration: Duration) {
        let mut rates = self.rates.lock().unwrap_or_else(|err| err.into_inner());
        if model.len() > 128 || (rates.len() >= 128 && !rates.contains_key(model)) {
            return;
        }
        let sample = rates.entry(model.to_owned()).or_default();
        sample.tokens += tokens;
        sample.duration += duration;
        sample.streams += 1;
    }

    pub(super) fn rates(
        &self,
        config: &Config,
        models: impl Iterator<Item = String>,
    ) -> HashMap<String, ModelRate> {
        let samples = self.rates.lock().unwrap_or_else(|err| err.into_inner());
        models
            .filter_map(|model| {
                let configured_tokens_per_second = config.model_profile(&model)?.tokens_per_second;
                let sample = samples.get(&model);
                let achieved_tokens_per_second = sample
                    .filter(|s| !s.duration.is_zero())
                    .map(|s| s.tokens as f64 / s.duration.as_secs_f64());
                let rate = ModelRate {
                    configured_tokens_per_second,
                    achieved_tokens_per_second,
                    completed_streams: sample.map_or(0, |s| s.streams),
                    output_tokens: sample.map_or(0, |s| s.tokens),
                };
                Some((model, rate))
            })
            .collect()
    }

    pub(super) fn substituted(&self) {
        self.shape_tool_substitutions
            .fetch_add(1, Ordering::Relaxed);
    }
}
