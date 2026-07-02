# Changelog

## [0.8.0] - 2026-07-01

> **Live validation:** thought-signature round trip confirmed end-to-end against live
> `gemini-3.1-pro-preview` (`TestLiveGemini3ToolCallThoughtSignatureRoundTrip`); without the fix the
> second turn is rejected with `400 INVALID_ARGUMENT: Function call is missing a thought_signature`.

### Added

- **Gemini 3 thought signatures.** `llm.ToolCallContent` and `llm.ToolCallStartEvent` gain an
  opaque `ThoughtSignature []byte` (additive, nil for providers without one). The Gemini provider
  captures `part.ThoughtSignature` from streamed function-call parts and replays it verbatim when
  history is converted back to genai contents — Gemini 3 rejects a replayed tool call without its
  signature, which broke every multi-turn tool loop on `gemini-3*` models. Callers that round-trip
  `StreamResult.Messages` get the fix for free; event-driven transcripts must carry the new event
  field onto the replayed `ToolCallContent`.

### Fixed

- **Tool calls no longer lost to chunk layout.** The Gemini stream loop only assembled
  `StreamResult.Messages` (and emitted `ToolCallEndEvent`) from function-call parts present in the
  same chunk as the finish reason; when the API delivered the call in an earlier chunk, the tool
  call vanished from the result (intermittent, observed live). Calls are now accumulated as parts
  arrive and flushed on finish, with a stream-end fallback.

## [0.7.0] - 2026-06-28

> **Live validation:** Mistral confirmed end-to-end against the live API; xAI and z.ai are
> dialect-verified (valid keys, request shapes accepted at the API boundary — full end-to-end
> pending account credits); Qwen is deferred (DashScope enterprise key). All four are covered by
> deterministic classifier and `httptest` body-capture tests.

### Added

- **xAI, Mistral, Qwen, and z.ai providers (LLM-10).** Four OpenAI-compatible targets ship as
  configured instances of the `openai-compat` adapter (no new provider type — ADR-0001), each with
  a constructor that fixes its base URL and thinking dialect so apps never hardcode endpoints:
  `openaicompat.XAI`, `Mistral`, `Qwen`, and `ZAI`, all taking a `TargetConfig` (env-sourced key via
  `APIKey`/`GetAPIKey`, plus optional headers). Per-model capabilities (thinking, vision) are
  resolved by a per-family classifier (ADR-0008, catalog-free), replacing the coarse
  `strings.HasPrefix(model, "o")` heuristic for these families. Thinking dialects per target: xAI and
  Mistral use `reasoning_effort`, gated per model because grok-4 / Magistral reason but reject the
  param (HTTP 400); Qwen uses the new top-level `enable_thinking` bool; z.ai/GLM reuses DeepSeek's
  `thinking:{type}` shape. Each target ships table-driven classifier tests, an `httptest`
  body-capture test for its request dialect, and an integration-gated live test keyed on
  `XAI_API_KEY` / `MISTRAL_API_KEY` / `DASHSCOPE_API_KEY` / `ZAI_API_KEY`.
- **`ThinkingQwen` thinking format.** New `openaicompat.ThinkingFormat` value emitting a top-level
  `enable_thinking` bool — the dialect Alibaba's DashScope compatible-mode endpoint uses for Qwen3,
  distinct from `ThinkingChatTemplate` (which nests the flag under `chat_template_kwargs`).
- **Named `openai-compat` providers (LLM-9).** `openaicompat.Config` gains a required `Name`
  field; `New` returns `ErrInvalidModel` when it is empty, and `Provider.Name()` reports it
  (replacing the hardcoded `"openai-compat"`). This lets multiple OpenAI-compatible targets
  (e.g. `xai`, `mistral`) coexist in the agent registry under distinct names instead of silently
  shadowing one another. Existing single-instance callers pass `Name: "openai-compat"`.

## [0.6.0] - 2026-06-26

### Changed

- **Module path changed to `github.com/dev-resolute/resolute-llm-go`** (was
  `github.com/resolute-sh/pi-llm-go`), part of the `resolute-sh`→`dev-resolute` rebrand. Update your
  import path: `go get github.com/dev-resolute/resolute-llm-go`. **No behaviour change** — pure
  module-path rename; the full test suite passes unchanged. ADR-0005 carries a rebrand note.

## [0.5.0] - 2026-06-26

### Added

- **`chat-template` thinking format (LLM-7).** `openaicompat.Compat.ThinkingFormat` gains
  `ThinkingChatTemplate`, which toggles reasoning via `chat_template_kwargs.enable_thinking`
  derived from the request's `ThinkingLevel`. For local reasoning models (Qwen3, DeepSeek-R1)
  served behind vLLM / llama.cpp where thinking is controlled by chat-template kwargs rather than
  `reasoning_effort`. Builds on the LLM-6 `Compat` seam (ADR-0008).
- **`ErrContextOverflow` detection (LLM-8).** New sentinel `llm.ErrContextOverflow` and
  `llm.AsContextOverflow(err)`, which wraps a provider error when its message reports the model's
  maximum context length was exceeded — matching both the `maximum context length of N tokens`
  and the parenthesized `maximum context length (N)` forms — so callers can react via `errors.Is`.
  Other errors and nil pass through unchanged. The detection half of the deferred auto-compaction
  story (ADR-0003).

## [0.4.0] - 2026-06-26

### Added

- **OpenAI-compat `Compat` config + `deepseek` thinking format (LLM-6).** `openaicompat.Config`
  gains a `Compat` field carrying caller-supplied per-model behaviour (ADR-0008, catalog-free):
  `ThinkingFormat` (`ThinkingReasoningEffort` default, `ThinkingDeepSeek`), `SupportsReasoningEffort`,
  `RequiresReasoningContentOnAssistantMessages`, and `MaxTokens`. Under `ThinkingDeepSeek`, thinking
  is toggled via `thinking:{type:"enabled"|"disabled"}` (plus `reasoning_effort` when supported),
  assistant messages can be required to carry a `reasoning_content` field, and `max_tokens` is sent
  when `MaxTokens > 0`. This adds DeepSeek V4 support on the opencode-go gateway
  (`https://opencode.ai/zen/go/v1`, `deepseek-v4-flash`/`deepseek-v4-pro`). The zero `Compat`
  preserves today's OpenAI behaviour exactly.

## [0.3.0] - 2026-06-26

### Added

- Gemini Vertex AI backend. `gemini.Config` gains `Vertex bool`, `Project`, and
  `Location`. When `Vertex` is set, the provider targets the Vertex AI backend via
  `google.golang.org/genai` and authenticates with Application Default Credentials
  (e.g. GKE Workload Identity) instead of an API key; `Project`/`Location` fall back
  to `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION`. The default (API key) path is
  unchanged.

### Fixed

- **Gemini 3 thinking now works (LLM-5).** Capabilities and the thinking-config mechanism
  are derived by generation (ADR-0008): Gemini 3.x / Gemma 4 use the `thinkingLevel` enum
  while Gemini 2.5 keeps `thinkingBudget` tokens. Previously a `Contains("2.5")` heuristic
  reported `Thinking:false`/`Vision:false` for `gemini-3.1-pro`, `gemini-3.5-flash`, and the
  `gemini-flash-latest` aliases, and the provider sent `thinkingBudget` (the wrong field) to
  Gemini 3 models.
- **Gemini thinking output now surfaces.** `IncludeThoughts` is set when thinking is on, so
  the model's reasoning is emitted as `ThinkingDeltaEvent`s; Gemini 3's "off" path uses the
  lowest supported level without `includeThoughts`.
- **Gemini streamed text no longer corrupts.** The stream loop treated each delta chunk's
  text as cumulative and dropped any chunk no longer than the running maximum, garbling
  multi-chunk responses. Deltas are now emitted directly, with thoughts and answer text on
  separate channels.

## [0.2.0] - 2026-05-29

Additive release supporting pi-core-agent-go v0.2.0. No breaking changes; existing
callers compile and behave identically except for the documented `minimal` fix.

### Added

- `ThinkingMinimal` level, inserted between `ThinkingOff` and `ThinkingLow`. Existing
  levels keep their meaning and ordering; `ThinkingOff` remains the zero value.
- `LLMRequest.SessionID` — optional session identifier. The OpenAI-compatible adapter
  sends it as prompt-cache affinity headers (`session_id`, `x-client-request-id`,
  `x-session-affinity`) and as the `prompt_cache_key` body param, matching upstream Pi;
  the Gemini adapter ignores it.
- `LLMRequest.Transport` (`TransportAuto` default, `TransportSSE`, `TransportWebSocket`)
  — transport preference. `TransportWebSocket` returns `ErrTransportUnsupported` from
  today's HTTP/SSE-only providers; reserved for a future websocket provider.
- `LLMRequest.ThinkingBudgets` (`map[ThinkingLevel]int`) — per-level token-budget
  overrides. Gemini applies the budget to `thinking_budget`; the OpenAI-compatible
  adapter ignores it (`reasoning_effort` is categorical, not a token count).
- `ErrTransportUnsupported` sentinel.

### Fixed

- `ThinkingMinimal` now maps to `minimal` reasoning effort on the OpenAI-compatible
  adapter (o-series) and the smallest valid `thinking_budget` on Gemini, instead of
  being treated as `low`. (upstream 0.37.0)
