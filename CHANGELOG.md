# Changelog

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
