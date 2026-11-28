//! HTTP coverage for stateless turn playback and opt-in load profiles.

use axum::{
    body::{to_bytes, Body},
    http::{HeaderMap, Request, StatusCode},
    Router,
};
use llmsim::{
    cli::{build_router, build_state, Config},
    count_tokens_default,
    scenario::ScenarioSet,
    stats::new_shared_stats,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;

const RESPONSES: &str = "/openai/v1/responses";
const MESSAGES: &str = "/anthropic/v1/messages";
const CHAT: &str = "/openai/v1/chat/completions";
const MARKED: &str = "Look up the record [[llmsim:s=t2,t1,o24 seed=-42]] and summarize it.";

fn config(overrides: &str) -> Config {
    Config::from_toml(&format!("[latency]\nprofile = \"instant\"\n{overrides}")).unwrap()
}

fn router(config: Config) -> Router {
    build_router(Arc::new(build_state(config, new_shared_stats()).unwrap()))
}

async fn request(
    router: &Router,
    path: &str,
    attempt: u32,
    body: Option<&Value>,
) -> (StatusCode, HeaderMap, String) {
    let req = Request::builder()
        .method(if body.is_some() { "POST" } else { "GET" })
        .uri(path)
        .header("content-type", "application/json")
        .header("x-stainless-retry-count", attempt)
        .body(body.map_or_else(Body::empty, |body| Body::from(body.to_string())))
        .unwrap();
    let res = router.clone().oneshot(req).await.unwrap();
    let (parts, body) = res.into_parts();
    let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
    (
        parts.status,
        parts.headers,
        String::from_utf8(bytes.to_vec()).unwrap(),
    )
}

async fn post(router: &Router, path: &str, attempt: u32, body: &Value) -> (HeaderMap, String) {
    let (status, headers, data) = request(router, path, attempt, Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{data}");
    (headers, data)
}

fn events(wire: &str) -> Vec<Value> {
    wire.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {"key": {"type": "string", "enum": ["record"]}},
        "required": ["key"],
        "additionalProperties": false
    })
}

fn responses_request(input: &[Value]) -> Value {
    json!({
        "model": "gpt-5",
        "stream": true,
        "store": false,
        "input": input,
        "parallel_tool_calls": true,
        "reasoning": {"effort": "low", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "tools": [
            {"type": "function", "name": "search", "parameters": schema(), "strict": true},
            {"type": "function", "name": "lookup", "parameters": schema()}
        ]
    })
}

fn completed(events: &[Value]) -> &Value {
    &events
        .iter()
        .find(|event| event["type"] == "response.completed")
        .unwrap()["response"]
}

#[tokio::test]
async fn responses_accepts_agent_replay_and_runs_two_parallel_rounds_then_text() {
    let server = router(config(""));
    let mut input = vec![
        json!({"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Use the offered tools."}]}),
        json!({"role": "user", "content": [{"type": "input_text", "text": MARKED}]}),
    ];

    for (step, expected_calls) in [2, 1, 0].into_iter().enumerate() {
        let (headers, wire) = post(&server, RESPONSES, 0, &responses_request(&input)).await;
        assert_eq!(headers["x-llmsim-step"].to_str().unwrap(), step.to_string());
        let events = events(&wire);
        let result = completed(&events);
        let usage = &result["usage"];
        let input_tokens = usage["input_tokens"].as_u64().unwrap();
        let output_tokens = usage["output_tokens"].as_u64().unwrap();
        assert!(input_tokens > 0 && output_tokens > 0);
        assert_eq!(usage["total_tokens"], input_tokens + output_tokens);
        assert!(usage["input_tokens_details"]["cached_tokens"].is_u64());
        let items = result["output"].as_array().unwrap();
        if expected_calls == 0 {
            assert_eq!(output_tokens, 24);
            assert_eq!(items.len(), 1);
            let text = items[0]["content"][0]["text"].as_str().unwrap();
            let streamed: String = events
                .iter()
                .filter(|e| e["type"] == "response.output_text.delta")
                .map(|e| e["delta"].as_str().unwrap())
                .collect();
            assert_eq!(text, streamed);
            assert_eq!(count_tokens_default(text).unwrap(), output_tokens as usize);
            continue;
        }
        assert_eq!(items.len(), expected_calls);
        assert!(events
            .iter()
            .any(|e| e["type"] == "response.function_call_arguments.delta"));
        if step == 0 {
            assert_ne!(items[0]["name"], items[1]["name"]);
        }
        input.push(
            json!({"type": "reasoning", "id": "rs_replayed", "summary": [
            {"type": "summary_text", "text": "I will check the record."}
        ], "encrypted_content": "opaque"}),
        );
        input.push(
            json!({"type": "message", "id": "msg_replayed", "status": "completed",
            "role": "assistant", "content": [{"type": "output_text",
            "text": "Checking the record now.", "annotations": []}]}),
        );
        input.extend(items.iter().cloned());
        for (index, item) in items.iter().enumerate() {
            assert_eq!(item["type"], "function_call");
            assert!(matches!(
                item["name"].as_str().unwrap(),
                "lookup" | "search"
            ));
            let arguments: Value =
                serde_json::from_str(item["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(arguments, json!({"key": "record"}));
            let output = if index == 0 {
                json!("record found")
            } else {
                json!([{"type": "input_text", "text": "record found", "prompt_cache_breakpoint": {"mode": "explicit"}}])
            };
            input.push(json!({"type": "function_call_output", "call_id": item["call_id"], "output": output}));
        }
        input.push(json!({"type": "configuration_update", "reasoning": {"effort": "medium"}}));
        if step == 0 {
            input.push(json!({"role": "user", "content": "Continue using the prior results."}));
        }
    }
}

#[tokio::test]
async fn responses_accepts_plain_replay_without_a_marker() {
    let server = router(config(""));
    let input = [
        json!({"role": "user", "content": "Look up the record"}),
        json!({"type": "reasoning", "id": "rs_a", "summary": []}),
        json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking now."}]}),
        json!({"type": "web_search_call", "id": "ws_a", "status": "completed", "action": {"type": "search", "query": "record"}}),
        json!({"type": "function_call", "call_id": "call_a", "id": "fc_a", "name": "lookup", "arguments": "{\"key\":\"record\"}"}),
        json!({"type": "function_call_output", "call_id": "call_a", "output": [{"type": "input_text", "text": "OK"}]}),
    ];
    let (headers, wire) = post(&server, RESPONSES, 0, &responses_request(&input)).await;
    assert!(!headers.contains_key("x-llmsim-step"));
    let events = events(&wire);
    let usage = &completed(&events)["usage"];
    assert!(usage["input_tokens"].as_u64().unwrap() > 0);
    let text: String = events
        .iter()
        .filter(|e| e["type"] == "response.output_text.delta")
        .map(|e| e["delta"].as_str().unwrap())
        .collect();
    assert_eq!(
        usage["output_tokens"].as_u64().unwrap(),
        count_tokens_default(&text).unwrap() as u64
            + usage["output_tokens_details"]["reasoning_tokens"]
                .as_u64()
                .unwrap(),
    );
}

#[tokio::test]
async fn anthropic_replays_tool_blocks_and_reports_cache_and_output_usage() {
    let server = router(config(""));
    let mut messages = vec![json!({"role": "user", "content": [{"type": "text", "text": MARKED}]})];
    for (step, expected_calls) in [2, 1, 0].into_iter().enumerate() {
        let req = json!({
            "model": "claude-sonnet-4-5",
            "stream": true,
            "max_tokens": 128,
            "messages": messages,
            "system": [{"type": "text", "text": "Use the provided tools. ".repeat(400), "cache_control": {"type": "ephemeral"}}],
            "tools": [{"name": "lookup", "input_schema": schema()}]
        });
        let (headers, wire) = post(&server, MESSAGES, 0, &req).await;
        assert_eq!(headers["x-llmsim-step"].to_str().unwrap(), step.to_string());
        let events = events(&wire);
        assert_eq!(events.last().unwrap()["type"], "message_stop");
        let usage = &events[0]["message"]["usage"];
        if step == 0 {
            assert!(usage["cache_creation_input_tokens"].as_u64().unwrap() > 0);
        } else {
            assert!(usage["cache_read_input_tokens"].as_u64().unwrap() > 0);
        }
        let terminal = events
            .iter()
            .find(|e| e["type"] == "message_delta")
            .unwrap();
        let starts = events
            .iter()
            .filter(|e| e["content_block"]["type"] == "tool_use");
        let mut calls = Vec::new();
        let mut outputs = Vec::new();
        for start in starts {
            let mut call = start["content_block"].clone();
            let args: String = events
                .iter()
                .filter(|e| {
                    e["index"] == start["index"] && e["delta"]["type"] == "input_json_delta"
                })
                .map(|e| e["delta"]["partial_json"].as_str().unwrap())
                .collect();
            call["input"] = serde_json::from_str(&args).unwrap();
            assert_eq!(call["name"], "lookup");
            assert_eq!(call["input"], json!({"key": "record"}));
            outputs.push(
                json!({"type": "tool_result", "tool_use_id": call["id"], "content": "found"}),
            );
            calls.push(call);
        }
        assert_eq!(calls.len(), expected_calls);
        if expected_calls > 0 {
            assert_eq!(terminal["delta"]["stop_reason"], "tool_use");
            assert!(terminal["usage"]["output_tokens"].as_u64().unwrap() > 0);
            messages.push(json!({"role": "assistant", "content": calls}));
            messages.push(json!({"role": "user", "content": outputs}));
        } else {
            assert_eq!(terminal["delta"]["stop_reason"], "end_turn");
            assert_eq!(terminal["usage"]["output_tokens"], 24);
            let text: String = events
                .iter()
                .filter(|e| e["delta"]["type"] == "text_delta")
                .map(|e| e["delta"]["text"].as_str().unwrap())
                .collect();
            assert_eq!(count_tokens_default(&text).unwrap(), 24);
        }
    }
}

#[tokio::test]
async fn seeded_replicas_agree_on_faults_and_tool_calls() {
    let settings = config("[errors]\nseed = 901\nrate_limit_rate = 0.5");
    let first = router(settings.clone());
    let second = router(settings);
    let req = responses_request(&[json!({"role": "user", "content": MARKED})]);
    let mut faults = 0;
    let mut successes = 0;
    for retry in 0..20 {
        let (a, b) = tokio::join!(
            request(&first, RESPONSES, retry, Some(&req)),
            request(&second, RESPONSES, retry, Some(&req)),
        );
        assert_eq!(a.0, b.0);
        if a.0 == StatusCode::OK {
            successes += 1;
            assert_eq!(a.1["x-llmsim-attempt"].to_str().unwrap(), retry.to_string());
            let events_a = events(&a.2);
            let events_b = events(&b.2);
            let calls = |finished: &Value| {
                finished["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| json!([item["call_id"], item["name"], item["arguments"]]))
                    .collect::<Vec<_>>()
            };
            assert_eq!(calls(completed(&events_a)), calls(completed(&events_b)));
        } else {
            faults += 1;
            assert_eq!(a.0, StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(a.1["retry-after"], b.1["retry-after"]);
            assert_eq!(a.2, b.2);
        }
    }
    assert!(faults > 0 && successes > 0);
}

#[tokio::test]
async fn seeded_named_scenarios_take_retry_number_from_the_header() {
    let scenario = r#"{
        "name": "retry",
        "steps": [{
            "fail_first": [{"kind": "rate_limit", "retry_after_ms": 17}],
            "turn": {"type": "assistant", "text": "Recovered"}
        }]
    }"#;
    let replicas: Vec<_> = (0..2)
        .map(|_| {
            let state = build_state(config("[errors]\nseed = -90"), new_shared_stats())
                .unwrap()
                .with_scenarios(Arc::new(ScenarioSet::from_json(scenario).unwrap()));
            build_router(Arc::new(state))
        })
        .collect();
    let req = json!({"model": "gpt-4o", "input": "Hello [[llmsim:retry]]"});
    let (headers, _) = post(&replicas[0], RESPONSES, 1, &req).await;
    assert_eq!(headers["x-llmsim-attempt"], "1");
    for replica in &replicas {
        for _ in 0..2 {
            let (status, headers, _) = request(replica, RESPONSES, 0, Some(&req)).await;
            assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(headers["retry-after-ms"], "17");
        }
        let (_, body) = post(replica, RESPONSES, 1, &req).await;
        assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap()["output_text"],
            "Recovered"
        );
    }
}

#[tokio::test]
async fn chat_shape_respects_first_user_and_reports_tool_substitutions() {
    let server = router(config(
        "[response]\nshape_tool_allowlist = [\"other\"]\ntarget_tokens = 12",
    ));
    let chat_tool =
        json!({"type": "function", "function": {"name": "lookup", "parameters": schema()}});
    for tools in [json!([]), json!([chat_tool])] {
        let req = json!({"model": "gpt-4o", "messages": [{"role": "user", "content": MARKED}], "tools": tools});
        let (_, body) = post(&server, CHAT, 0, &req).await;
        let response: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(response["choices"][0]["finish_reason"], "stop");
        assert_eq!(response["usage"]["completion_tokens"], 12);
    }
    let req = json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Ordinary request"},
            {"role": "user", "content": MARKED}
        ]
    });
    let (headers, _) = post(&server, CHAT, 0, &req).await;
    assert!(!headers.contains_key("x-llmsim-step"));
    let (_, _, data) = request(&server, "/llmsim/stats", 0, None).await;
    assert_eq!(
        serde_json::from_str::<Value>(&data).unwrap()["shape_tool_substitutions"],
        2
    );
}

#[tokio::test]
async fn bytes_estimate_and_model_rates_are_reported() {
    let config = Config::from_toml(
        r#"
token_estimate = "bytes"
[latency]
profile = "instant"
[model_profiles."gpt-*"]
ttft_ms = 0
tokens_per_second = 200
output_tokens = [60, 60]
[model_profiles."gpt-4o"]
ttft_ms = 0
tokens_per_second = 240
output_tokens = 60
"#,
    )
    .unwrap();
    assert!(Config::from_toml(include_str!("../examples/load-test-profile.toml")).is_ok());
    let server = router(config);
    let prompt = "😀".repeat(20);
    let req = json!({"model": "gpt-4o", "stream": true, "input": prompt});
    let (_, wire) = post(&server, RESPONSES, 0, &req).await;
    let events = events(&wire);
    let usage = &completed(&events)["usage"];
    assert_eq!(usage["input_tokens"], 20);
    assert_eq!(usage["output_tokens"], 60);
    let (_, _, stats) = request(&server, "/llmsim/stats", 0, None).await;
    let stats: Value = serde_json::from_str(&stats).unwrap();
    let rate = &stats["model_output_rates"]["gpt-4o"];
    assert_eq!(rate["configured_tokens_per_second"], 240.0);
    assert_eq!(rate["completed_streams"], 1);
    assert_eq!(rate["output_tokens"], usage["output_tokens"]);
    let achieved = rate["achieved_tokens_per_second"].as_f64().unwrap();
    assert!(achieved > 0.0 && achieved <= 250.0, "{achieved}");

    let req = json!({"model": "unknown-model", "input": prompt});
    let (_, body) = post(&server, RESPONSES, 0, &req).await;
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["usage"]["input_tokens"],
        20
    );
    let (_, _, stats) = request(&server, "/llmsim/stats", 0, None).await;
    assert!(
        serde_json::from_str::<Value>(&stats).unwrap()["model_output_rates"]["unknown-model"]
            .is_null()
    );
}

#[tokio::test]
async fn responses_reports_cache_writes_and_reads_for_current_models() {
    let config = Config::from_toml(
        r#"
token_estimate = "bytes"
[latency]
profile = "instant"
[response]
generator = "fixed:Done"
"#,
    )
    .unwrap();
    let server = router(config);
    let req = json!({
        "model": "gpt-5.6",
        "input": "read ".repeat(1200),
        "store": false
    });
    let (_, first) = post(&server, RESPONSES, 0, &req).await;
    let (_, second) = post(&server, RESPONSES, 0, &req).await;
    let first: Value = serde_json::from_str(&first).unwrap();
    let second: Value = serde_json::from_str(&second).unwrap();
    let first = &first["usage"]["input_tokens_details"];
    let second = &second["usage"]["input_tokens_details"];
    assert_eq!(first["cached_tokens"], 0);
    assert!(first["cache_write_tokens"].as_u64().unwrap() > 0);
    assert_eq!(second["cached_tokens"], first["cache_write_tokens"]);
    assert_eq!(second["cache_write_tokens"], 0);
}

#[tokio::test]
async fn named_replay_uses_latest_turn_and_shapes_output_with_tools() {
    let server = router(config(
        "[response]\nshape_tool_allowlist = [\"web_search\", \"url_scraper\"]",
    ));
    let mut input = vec![
        json!({"role": "user", "content": "[[llmsim:s=v2,o7 seed=2]]"}),
        json!({"role": "assistant", "content": "Previous answer"}),
        json!({"role": "user", "content": "[[llmsim:s=v2,tw1+u1@120,o40 seed=3]]"}),
    ];
    for step in 0..2 {
        let mut req = responses_request(&input);
        req["tools"] = json!([
            {"type": "function", "name": "web_search"},
            {"type": "function", "name": "url_scraper"}
        ]);
        let (headers, wire) = post(&server, RESPONSES, 0, &req).await;
        assert_eq!(headers["x-llmsim-step"].to_str().unwrap(), step.to_string());
        let items = events(&wire);
        let result = completed(&items);
        let output = result["output"].as_array().unwrap();
        let expected_tokens = if step == 0 { 120 } else { 40 };
        assert_eq!(result["usage"]["output_tokens"], expected_tokens);
        let names: Vec<_> = output
            .iter()
            .filter_map(|item| item["name"].as_str())
            .collect();
        if step == 0 {
            assert_eq!(names, ["web_search", "url_scraper"]);
            for item in output.iter().filter(|item| item["type"] == "function_call") {
                input.push(json!({"type": "function_call", "name": item["name"],
                    "call_id": item["call_id"], "arguments": item["arguments"]}));
            }
            for item in output.iter().filter(|item| item["type"] == "function_call") {
                input.push(json!({"type": "function_call_output",
                    "call_id": item["call_id"], "output": "Synthetic result"}));
            }
        } else {
            assert!(names.is_empty());
        }
    }
    let (_, _, stats) = request(&server, "/llmsim/stats", 0, None).await;
    let stats: Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(stats["shape_requests"], 2);
    assert_eq!(stats["shape_tool_calls"], 2);
    assert_eq!(stats["shape_output_tokens"], 160);
    assert_eq!(stats["shape_tool_substitutions"], 0);
}

#[tokio::test]
async fn named_replay_loads_deferred_toolkit_without_consuming_scripted_round() {
    let server = router(config(
        "[response]\nshape_tool_allowlist = [\"exec_command\"]",
    ));
    let mut input = vec![json!({
        "role": "user", "content": "[[llmsim:s=v2,te1@100,o20 seed=22]]"
    })];
    let mut req = responses_request(&input);
    req["tools"] = json!([{"type": "function", "name": "activate_toolkit"}]);
    let (headers, wire) = post(&server, RESPONSES, 0, &req).await;
    assert_eq!(headers["x-llmsim-step"], "0");
    let items = events(&wire);
    let activation = &completed(&items)["output"][0];
    assert_eq!(activation["name"], "activate_toolkit");
    assert_eq!(
        activation["arguments"],
        r#"{"toolkit_name":"code_execution"}"#
    );
    input.push(json!({"type": "function_call", "name": activation["name"],
        "arguments": activation["arguments"], "call_id": activation["call_id"]}));
    input.push(json!({"type": "function_call_output",
        "output": "Toolkit loaded", "call_id": activation["call_id"]}));
    let mut req = responses_request(&input);
    req["tools"] = json!([{"type": "function", "name": "exec_command"}]);
    let (headers, wire) = post(&server, RESPONSES, 0, &req).await;
    assert_eq!(headers["x-llmsim-step"], "0");
    let items = events(&wire);
    let output = completed(&items)["output"].as_array().unwrap();
    assert!(output.iter().any(|item| item["name"] == "exec_command"));
    let (_, _, stats) = request(&server, "/llmsim/stats", 0, None).await;
    let stats: Value = serde_json::from_str(&stats).unwrap();
    assert_eq!(stats["shape_requests"], 2);
    assert_eq!(stats["shape_activation_requests"], 1);
    assert_eq!(stats["shape_tool_substitutions"], 0);
}
