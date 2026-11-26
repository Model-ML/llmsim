---
type: Specification
title: Stateless Turn Shapes and Load Profiles
description: Replay portable agent turns and measure output pacing against configurable model profiles.
tags:
  - llmsim
  - simulation
  - load-testing
---
# Stateless Turn Shapes and Load Profiles

## Abstract

Load tests need agent conversations that behave consistently when requests
land on independent replicas. Inline turn shapes derive work from the
submitted history. Model profiles control timing and default length, and
attempt-keyed faults let a retry move between replicas.

## Requirements

**R1**. HTTP Responses, Anthropic Messages, and Chat Completions recognize
`[[llmsim:s=<shape> seed=<signed-i64>]]` in the first user message. The shape
grammar is `(tN,)*oN`, where each positive `tN` is a tool-call round and the
positive `oN` is the final output length. Count replayed assistant rounds
with tool calls, including parallel calls as one round. Keep this resolution
independent of request order and server state.

**R2**. Hash the marker payload and seed into the session identity. Pick and
reuse only offered functions permitted by the optional allowlist. Produce
arguments using the shared JSON Schema generator, and record a text
substitution when no eligible tool is offered. Render normal terminal usage
in the provider's wire format.

**R3**. Resolve optional glob-based model profiles for TTFT, output rate,
and default output length. Fall back to existing defaults when no glob
matches. Report the achieved rate for completed, profile-paced streams so
load tests can detect drift against the configured rate. Scheduled output
uses one clock per stream to avoid accumulating timer rounding at each delta.

**R4**. A configured run seed makes random faults and named scenario retries
reproducible for explicit history. Read the SDK's zero-based retry header
on each request. Keep existing unseeded fault behavior. Stored-response
continuation retains its server-local history dependency.

**R5**. Provide an opt-in byte estimate for input accounting and keep the
tokenizer as the default. Usage includes available cache read and write
fields and accounts for the completed output.

See the [operator guide](../../docs/load-test-profiles.md) for configuration,
marker limits, and the metrics contract.
