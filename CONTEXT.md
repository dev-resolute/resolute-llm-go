# pi-llm-go — Glossary

## LLM Layer Terms

### Provider model

**LLMProvider**: The single interface every concrete provider implements. Exposes streaming, tool calls, and thinking blocks in a normalized event shape.

**OpenAI-compatible adapter**: One `LLMProvider` implementation, parameterized by base URL, covering OpenAI, Fireworks, OpenCode Zen, the hosted providers xAI/Mistral/Qwen/z.ai, and locally-hosted servers (Ollama, vLLM, llama.cpp server, LM Studio). The same code is configured N times under distinct `Name`s. Tool call IDs are uniquified at the conversion boundary (empty IDs get `call_N`, duplicates get occurrence suffixes) so cross-provider transcripts replay with unambiguous call/result pairing.

**Config.Name**: Required identifier on `openaicompat.Config` (LLM-9), used as the registry key and in `<name>/<model>` refs. `New` rejects an empty `Name`, so multiple compat targets coexist instead of shadowing one another.

**Provider constructors**: `openaicompat.XAI`, `Mistral`, `Qwen`, `ZAI` (LLM-10) — each fixes its base URL, thinking dialect, and per-family capability classifier, taking only a `TargetConfig` (env-sourced key plus optional headers) so apps never hardcode endpoints.

**Gemini provider**: The `LLMProvider` implementation built on `google.golang.org/genai`.

**GetAPIKey**: Optional `func(ctx) (string, error)` on each provider's `Config`, resolved once per `Stream` call for short-lived credentials; falls back to the static `APIKey` when nil.

**Config.Headers**: Static per-provider headers merged into every request, overridden by per-call `LLMRequest.Headers` on conflict.

### Streaming

**EventStream**: Struct with `Events <-chan LLMEvent` and `Done <-chan StreamResult`. The shared return shape for any LLM streaming call.

**StreamResult**: Terminal value on `EventStream.Done`. Contains final messages + error.

**LLMEvent**: Sealed interface for events on `EventStream.Events`. Concrete variants: `TextDeltaEvent`, `ThinkingDeltaEvent`, `ToolCallStartEvent`, `ToolCallEndEvent`, `MessageEndEvent`, `LLMErrorEvent`, `LLMRetryEvent`, `UsageEvent`.

### Request

**LLMRequest**: Inputs for a single streaming call — `Model`, `Messages`, `Tools`, `Thinking`, plus the per-call fields below.

**LLMRequest.Headers**: Per-call header overrides merged over `Config.Headers` (request wins). The path for hook-injected trace/tenant headers to reach the wire.

**SessionID**: Optional conversation identifier for prompt-cache affinity. The OpenAI-compatible adapter sends it as affinity headers (`session_id`, `x-client-request-id`, `x-session-affinity`) and the `prompt_cache_key` body param; the Gemini adapter ignores it.

**Transport**: `TransportPreference` enum (`TransportAuto` default, `TransportSSE`, `TransportWebSocket`). `TransportWebSocket` returns `ErrTransportUnsupported` from today's HTTP/SSE-only providers; reserved for a future websocket provider.

**OnBeforeRequest / OnAfterResponse**: Per-call hooks on `LLMRequest`. `OnBeforeRequest(headers)` may mutate the merged headers before send; `OnAfterResponse(statusCode, headers)` observes response metadata.

### Messages

**Message**: LLM-side unit of content. Struct with `Role` and sealed `Content` interface. Distinct from agent-side transcript `Message`.

**Content**: Sealed interface with variants `TextContent`, `ToolCallContent`, `ToolResultContent`, `ThinkingContent`.

**ToolDef**: LLM-visible tool spec. `Name`, `Description`, `Schema json.RawMessage`.

### Thinking and capabilities

**ThinkingLevel**: Portable enum (`ThinkingOff`, `ThinkingMinimal`, `ThinkingLow`, `ThinkingMedium`, `ThinkingHigh`). `ThinkingOff` is the zero value; `ThinkingMinimal` sits between `ThinkingOff` and `ThinkingLow`, mapping to `minimal` reasoning effort on the OpenAI-compatible adapter and the smallest valid `thinking_budget` on Gemini.

**ProviderHints**: Typed escape hatch with nilable typed pointers (`*OpenAIHints`, `*GeminiHints`).

**ThinkingBudgets**: Optional `map[ThinkingLevel]int` on `LLMRequest` overriding the per-level token budget. Gemini applies it to `thinking_budget`; the OpenAI-compatible adapter ignores it (`reasoning_effort` is categorical). `ProviderHints` takes precedence when both are set.

**ProviderCapabilities**: Concrete bool fields per model: `Streaming`, `ToolCalling`, `ParallelToolCalls`, `Thinking`, `PromptCaching`, `Vision`.

**Gemini model class**: How the Gemini provider derives per-model behaviour from the bare model string, with no model catalog (ADR-0001). Classified by generation: Gemini 2.5 drives reasoning via `thinkingBudget` (a token count); Gemini 3.x (`gemini-3*-pro`, `gemini-3*-flash`, `gemini-flash-latest`, `gemini-flash-lite-latest`) and Gemma 4 drive it via `thinkingLevel` (enum `MINIMAL`/`LOW`/`MEDIUM`/`HIGH`). The same classification feeds `ProviderCapabilities` (`Thinking` and `Vision` per generation) and the thinking-off path (Gemini 3 cannot fully disable thinking — it gets the lowest level without `includeThoughts`). Mirrors upstream `google.ts`'s `isGemini3ProModel`/`isGemini3FlashModel`/`isGemma4Model` predicates but stays catalog-free.
_Avoid_: model catalog, model registry (we classify by generation, not a metadata table).

**Thought signature**: An opaque byte token Gemini 3 attaches to streamed function-call parts and requires back, verbatim, on the same tool call when the history is replayed — a missing signature rejects the whole turn (`400 INVALID_ARGUMENT`), killing multi-turn tool loops. Carried as `ThoughtSignature []byte` on `ToolCallContent` (replayed by `toGeminiContents`) and on `ToolCallStartEvent` (so event-driven transcripts can persist it); nil everywhere else — no provider fabricates one. Mirrors upstream `google.ts`'s thoughtSignature handling.
_Avoid_: reasoning signature, thought token.

**Compat model class**: The OpenAI-compatible analog of the Gemini model class, for the built-in compat providers (LLM-10, ADR-0008). A catalog-free classifier per family (`classifyXAI`, `classifyMistral`, `classifyQwen`, `classifyZAI`) maps a bare model id to its `Thinking`/`Vision` capabilities and whether it accepts `reasoning_effort` — the last bit gates the param because some models reason but reject it (xAI grok-4, Mistral Magistral → HTTP 400). Replaces the coarse `HasPrefix(model, "o")` heuristic for these families; a plain `New` keeps the heuristic.

**Compat**: Per-model behaviour config on the OpenAI-compatible provider, supplied by the caller (we ship no model catalog — ADR-0001). Mirrors the upstream `compat` object. Carries `ThinkingFormat`, `SupportsReasoningEffort`, `RequiresReasoningContentOnAssistantMessages` (echo prior plain `reasoning_content` back to the model on assistant messages), and `MaxTokensField` (`max_tokens` vs `max_completion_tokens`). The caller sets the `Compat` matching their model when it deviates from the OpenAI default.
_Avoid_: ModelCompat, Quirks

**ThinkingFormat**: The wire mechanism an OpenAI-compatible model uses to toggle reasoning, selected by `Compat.ThinkingFormat`: `reasoning_effort` (OpenAI o-series + xAI/Mistral — the default; gated per model by the compat model class), `deepseek` (`thinking:{type:enabled/disabled}` plus optional `reasoning_effort`; DeepSeek V4 on the opencode-go gateway, reused by z.ai/GLM), `chat-template` (`chat_template_kwargs.enable_thinking`, for Qwen/DeepSeek behind vLLM), and `qwen` (top-level `enable_thinking` bool, for Alibaba DashScope). Adding a new format is a new enum value plus one branch, never a catalog.
_Avoid_: ReasoningMode, ThinkingMechanism

### Retry

**RetryPolicy**: Shared config shape. `MaxRetries`, `MaxRetryDelay`. Delegated to underlying SDK.

**LLMRetryEvent**: Emitted per retry attempt for observability.

### Testing

**MockProvider**: Fluent-builder `LLMProvider` in `mock/` subpackage. Scripts responses keyed by matcher.
