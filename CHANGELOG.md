# Changelog

## [Unreleased]

### Added

- Gemini Vertex AI backend. `gemini.Config` gains `Vertex bool`, `Project`, and
  `Location`. When `Vertex` is set, the provider targets the Vertex AI backend via
  `google.golang.org/genai` and authenticates with Application Default Credentials
  (e.g. GKE Workload Identity) instead of an API key; `Project`/`Location` fall back
  to `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION`. The default (API key) path is
  unchanged.

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
