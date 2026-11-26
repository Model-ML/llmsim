# Multi-turn load tests

Start LLMSim with the [sample profile](../examples/load-test-profile.toml).

```sh
RUST_LOG=warn llmsim serve --config examples/load-test-profile.toml
```

## Inline turn shapes

Put a marker anywhere in the text of the first user message, such as
`Find the record [[llmsim:s=t2,t1,o300 seed=42]] and summarize it.` The
comma-separated shape has zero or more `tN` steps followed by one `oN` step.
This example requests two parallel calls on the first request, one call on
the second, and a final answer of about 300 tokens. The seed is a signed
64-bit integer. A shape supports up to 128 tool rounds, 128 calls per round,
and 32,768 output tokens. The request's output cap limits the final text.

Send the original user message and the tool-call history on each request.
Responses counts a group of assistant `function_call` items as one round,
Anthropic counts an assistant message containing `tool_use` blocks, and
Chat Completions counts an assistant message with `tool_calls`. Replay the
matching results as `function_call_output`, `tool_result`, or tool-role
messages, respectively. Later user messages can add context without moving
the marker. Requests after the last tool round return the final text again.

The marker and seed identify a session. A session and step select the same
function names, call IDs, arguments, and final text on each replica. Calls
cycle through a deterministic ordering of the offered functions when the
shape asks for more calls than there are tools. Configure
`[response].shape_tool_allowlist` to limit playback to specific function
names. An empty list permits all offered functions. When no eligible tools
are available, the tool step answers in text using `[response].target_tokens`
and increments `shape_tool_substitutions` in `GET /llmsim/stats`.

The inline marker takes precedence over named scenarios and scripts on these
three HTTP endpoints. Use [named scenarios](scenarios.md) for per-step stalls,
stream cuts, and scripted failures. With the Responses API, send the full
history with `store: false` for round-robin tests. Continuation by
`previous_response_id` requires the replica that stored the parent.

## Model profiles and usage

Add a `[model_profiles."<glob>"]` table with `ttft_ms`, `tokens_per_second`,
and `output_tokens`. TTFT and output length accept a fixed integer, an
inclusive range such as `[650, 1100]`, or a log-normal distribution specified
as `{ median = 850, p99 = 2200 }`. The `*` wildcard matches any number of
characters and `?` matches one. Exact names win, then the most literal
characters, fewest wildcards, and alphabetically first glob. Models without a
match use the existing `[latency]` settings or built-in model timings and
`[response].target_tokens`. Profile TTFT includes the simulated startup wait.
Set `[latency].prefill_us_per_token` to add prompt-size-dependent prefill.

The output-length distribution applies to the ordinary length-based
generators. Inline shapes, named scenarios, scripts, fixed or echo text,
structured output, and tool calls choose their own output length. Named
scenarios can also override pacing.

For each observed model with a profile, `GET /llmsim/stats` reports
`configured_tokens_per_second`, `achieved_tokens_per_second`,
`completed_streams`, and `output_tokens` in `model_output_rates`. The
achieved rate is the sum of measured output tokens divided by the sum of
their generation durations. It measures server-paced output after TTFT and
excludes hidden reasoning, interrupted streams, and scenario steps with
their own rate, speed multiplier, or stall. An achieved rate is `null`
until a sample completes. For a 5% drift tolerance compare
`abs(achieved / configured - 1)` with `0.05`.

Set the top-level `token_estimate = "bytes"` to estimate prompt text at one
token per four UTF-8 bytes. Request framing and supported input-image
estimates still apply. The default is `"tiktoken"`. The estimate is used for
input validation, usage, prompt caching, and prefill on the three generation
endpoints, and for Anthropic's count-tokens endpoint. Output usage is counted
separately. OpenAI reports cached reads in input token details and estimates
writes for GPT-5.6 and later at the simulator's existing implicit prefix
boundaries. Responses includes reasoning in total output tokens. Anthropic
reports cache reads and writes separately from uncached input tokens.

## Seeded faults and logging

Set `[errors].seed` to reproduce configured fault draws across replicas.
The draw uses the run seed, session, step, and zero-based
`x-stainless-retry-count` header. A missing or invalid header means attempt
zero. Unmarked requests use a hash of the first user text for their session.
Named scenarios use their existing step identity and the same retry header
for `fail_first`, `error_rate`, and stream cuts. With a run seed, configured
overload bursts on these endpoints become independent per-attempt overloads
with the configured retry delay. Explicit per-key quotas and prompt caches
retain their own server-local state.

Without an error seed, configured random faults and named scenario attempt
tracking use their existing behavior. Inline shapes always use the retry
header for their attempt number. Set `RUST_LOG=warn` or `RUST_LOG=off` to
silence request logs in server mode. With no `RUST_LOG`, the default logging
levels apply.
