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

**LLMEvent**: Sealed interface for events on `EventStream.Events`. Concrete variants: `TextDeltaEvent`, `ThinkingDeltaEvent`, `ToolCallStartEvent`, `ToolCallEndEvent`, `MessageEndEvent`, `LLMErrorEvent`, `LLMRetryEvent`, `UsageEvent`. Providers emit **at most one `UsageEvent` per stream**, carrying the request's final totals (`InputTokens` = prompt − cache-read − cache-write; `OutputTokens` = completion, including reasoning) immediately before `MessageEndEvent`, so consumer-side accumulation is exactly-once per call; last report wins when several chunks carry usage.

**StopReason** (LLM-15): String enum on `MessageEndEvent` describing why the assistant message ended — `StopReasonStop`, `StopReasonLength`, `StopReasonToolUse`, and `StopReasonUnknown` (the zero value; not emitted by either shipped provider on a successful stream since LLM-17). Mirrors upstream's stop-reason set minus error/aborted/pending/deferred: terminal provider stops are fatal errors (see **Provider stop**, LLM-17), upstream's pending is an interim state this event shape does not expose, and deferred responses are out of scope. Length wins over tool use, since a truncated message's tool-call arguments may be incomplete (upstream #6285) and callers need to be able to tell.

### Request

**LLMRequest**: Inputs for a single streaming call — `Model`, `Messages`, `Tools`, `Thinking`, plus the per-call fields below.

**LLMRequest.Headers**: Per-call header overrides merged over `Config.Headers` (request wins). The path for hook-injected trace/tenant headers to reach the wire.

**SessionID**: Optional conversation identifier for prompt-cache affinity. The OpenAI-compatible adapter sends it as affinity headers (`session_id`, `x-client-request-id`, `x-session-affinity`) and the `prompt_cache_key` body param; the Gemini adapter ignores it.

**Transport**: `TransportPreference` enum (`TransportAuto` default, `TransportSSE`, `TransportWebSocket`). `TransportWebSocket` returns `ErrTransportUnsupported` from today's HTTP/SSE-only providers; reserved for a future websocket provider.

**OnBeforeRequest / OnAfterResponse**: Per-call hooks on `LLMRequest`. `OnBeforeRequest(headers)` may mutate the merged headers before send; `OnAfterResponse(statusCode, headers)` observes response metadata.

### Messages

**Message**: LLM-side unit of content. Struct with `Role` and sealed `Content` interface. Distinct from agent-side transcript `Message`.

**Content**: Sealed interface with variants `TextContent`, `ToolCallContent`, `ToolResultContent`, `ThinkingContent`.

**ImageContent**: Sealed content type carrying raw image bytes and MIME type. Valid as part of a user message (a text + image turn is two adjacent user messages) and on the new `ToolResultContent.Images` field. Gemini sends images as `inlineData` parts on plain user messages. Tool-result encoding on Gemini (upstream `google-shared.ts` parity): the `functionResponse` body keys on `IsError` — `{"output": ...}` on success, `{"error": ...}` when `ToolResultContent.IsError` is set — and consecutive tool-result messages merge into a single `user` turn carrying multiple `functionResponse` parts instead of one turn per result; tool-result *images* route by generation — Gemini ≥3 and Gemma 4 nest them inside that same `functionResponse` part (`FunctionResponsePart.InlineData`), while Gemini 2.5 and legacy models fall back to a separate trailing user turn carrying the literal `"Tool result image:"` marker text (upstream's universally-supported google-shared layout, now the pre-3 fallback rather than the only path). The OpenAI-compatible adapter sends `image_url` with data URLs and batches consecutive tool-result images into one trailing user message whose content array leads with a `"Attached image(s) from tool result:"` text part followed by the batched `image_url` parts (upstream `openai-completions.ts` parity); the tool message's own `content` field gets `"(see attached image)"` substituted in when its text is empty and images are attached (three-way placeholder: text kept when non-empty, `"(see attached image)"` when empty with images, `"(no tool output)"` when empty without).

**ToolDef**: LLM-visible tool spec. `Name`, `Description`, `Schema json.RawMessage`, `ConstrainedSampling`.

**ConstrainedSampling** (LLM-12): Optional `*ConstrainedSampling` field on `ToolDef` opting a tool into provider-side JSON-schema-enforced ("strict") sampling; nil (the default) is off, matching upstream's omitted/false. Carries a single `Strict StrictMode` field, resolved by the one `ResolveStrictSampling(tool, supported)` function both adapters call — no per-provider resolution logic. Whether `supported` is true is provider/model-specific: the openai-compat adapter gates per instance via `Config.SupportsStrictTools` (nil defers to the family classifier's `strictTools` default, true for a plain instance and every current named family), emitting a `"strict"` key on every function tool when supported and omitting the key entirely when not; the Gemini adapter gates by generation (`class3Pro`/`class3Flash` only — Gemini 2.5, legacy, and Gemma 4 are unsupported since upstream's `^gemini-` gate excludes Gemma) and, when any tool resolves strict, sets `ToolConfig.FunctionCallingConfig.Mode = VALIDATED` for the whole request (upstream `google-shared.ts:311-324` — request-level, not per-tool). Mirrors upstream `constrained-sampling.ts`'s json_schema half; the grammar half (`openai_lark`/`openai_regex`, `type:"custom"` tools) is a deliberate watch item, not ported — no chat-completions model in upstream's catalog gets grammar sampling, and today's adapters here are both chat-completions-shaped.

**StrictMode** (LLM-12): `"prefer"` | `"require"` string enum on `ConstrainedSampling.Strict`. `StrictPrefer` degrades silently — strict sampling is used when supported, and quietly skipped (no error, no wire field) when not. `StrictRequire` is a hard requirement: when unsupported, `ResolveStrictSampling` returns an error naming the tool (`Tool "<name>" requires JSON-schema constrained sampling, but strict tools are unsupported.`), which both adapters surface as a fatal pre-flight error wrapped in `llm.ErrProviderFatal` — LLM-11 polarity, since it's a deterministic config error that will fail identically on retry. A non-nil `ConstrainedSampling` whose `Strict` is any other value (including the zero value `""`) is itself a resolver error naming the invalid value.

### Thinking and capabilities

**ThinkingLevel**: Portable enum (`ThinkingOff`, `ThinkingMinimal`, `ThinkingLow`, `ThinkingMedium`, `ThinkingHigh`, `ThinkingXhigh`, `ThinkingMax`). `ThinkingOff` is the zero value; `ThinkingMinimal` sits between `ThinkingOff` and `ThinkingLow`, mapping to `minimal` reasoning effort on the OpenAI-compatible adapter and the smallest valid `thinking_budget` on Gemini. `ThinkingXhigh`/`ThinkingMax` (LLM-13, upstream 0.80.6+) sit above `ThinkingHigh`: the OpenAI-compatible adapter passes them straight through as `reasoning_effort: "xhigh"`/`"max"` (models that don't support the extra levels just see an effort string they ignore or reject, same as any unsupported value today); Gemini has no levels above `HIGH`, so both clamp to `genai.ThinkingLevelHigh` with the same default token budget as `ThinkingHigh` (still overridable via `req.ThinkingBudgets`).

**ProviderHints**: Typed escape hatch with nilable typed pointers (`*OpenAIHints`, `*GeminiHints`).

**ThinkingBudgets**: Optional `map[ThinkingLevel]int` on `LLMRequest` overriding the per-level token budget. Gemini applies it to `thinking_budget`; the OpenAI-compatible adapter ignores it (`reasoning_effort` is categorical). `ProviderHints` takes precedence when both are set.

**ProviderCapabilities**: Concrete bool fields per model: `Streaming`, `ToolCalling`, `ParallelToolCalls`, `Thinking`, `PromptCaching`, `Vision`.

**Gemini model class**: How the Gemini provider derives per-model behaviour from the bare model string, with no model catalog (ADR-0001). Classified by generation: Gemini 2.5 drives reasoning via `thinkingBudget` (a token count); Gemini 3.x (`gemini-3*-pro`, `gemini-3*-flash`, `gemini-flash-latest`, `gemini-flash-lite-latest`) and Gemma 4 drive it via `thinkingLevel` (enum `MINIMAL`/`LOW`/`MEDIUM`/`HIGH`). The same classification feeds `ProviderCapabilities` (`Thinking` and `Vision` per generation) and the thinking-off path (Gemini 3 cannot fully disable thinking — it gets the lowest level without `includeThoughts`). Mirrors upstream `google.ts`'s `isGemini3ProModel`/`isGemini3FlashModel`/`isGemma4Model` predicates but stays catalog-free.
_Avoid_: model catalog, model registry (we classify by generation, not a metadata table).

**Thought signature**: An opaque byte token Gemini attaches to streamed parts — function-call, text, and thinking — and requires back, verbatim, when the history is replayed. A missing function-call signature rejects the whole turn (`400 INVALID_ARGUMENT`), killing multi-turn tool loops; dropping a *signed empty* text/thinking part breaks the reasoning chain and the model intermittently ends mid-task turns with a thought-only STOP (upstream #7362). Carried as `ThoughtSignature []byte` on `ToolCallContent`, `TextContent`, and `ThinkingContent` (replayed by `toGeminiContents`, which drops empty assistant text/thinking blocks only when they are unsigned), and on `ToolCallStartEvent`/`TextDeltaEvent`/`ThinkingDeltaEvent` so event-driven transcripts can persist it — a signature typically arrives on one delta of a part (possibly one with an empty `Delta`), so consumers retain the last non-empty value (upstream `retainThoughtSignature`). Nil everywhere else — no provider fabricates one. Mirrors upstream `google-shared.ts`'s thoughtSignature handling.
_Avoid_: reasoning signature, thought token.

**Tool-call ID replay**: On Gemini 3+ (`class3Pro`/`class3Flash`), `toGeminiContents` sets `FunctionCall.ID`/`FunctionResponse.ID` from `ToolCallContent.CallID`/`ToolResultContent.CallID` — signed multi-turn replay breaks without them (upstream #7494, `requiresToolCallId`). Pre-Gemini-3 models and Gemma 4 get no IDs (their backends reject or ignore them).

**Compat model class**: The OpenAI-compatible analog of the Gemini model class, for the built-in compat providers (LLM-10, ADR-0008). A catalog-free classifier per family (`classifyXAI`, `classifyMistral`, `classifyQwen`, `classifyZAI`) maps a bare model id to its `Thinking`/`Vision` capabilities and whether it accepts `reasoning_effort` — the last bit gates the param because some models reason but reject it (xAI grok-4, Mistral Magistral → HTTP 400). Replaces the coarse `HasPrefix(model, "o")` heuristic for these families; a plain `New` keeps the heuristic.

**Compat**: Per-model behaviour config on the OpenAI-compatible provider, supplied by the caller (we ship no model catalog — ADR-0001). Mirrors the upstream `compat` object. Carries `ThinkingFormat`, `SupportsReasoningEffort`, `RequiresReasoningContentOnAssistantMessages` (echo prior plain `reasoning_content` back to the model on assistant messages), `MaxTokensField` (`max_tokens` vs `max_completion_tokens`), and `SupportsFinishReason` (nil = the provider is expected to terminate streams with `finish_reason`, so a stream ending without one is a protocol error; set false for providers known to omit it — some vLLM/llama.cpp-style local servers — and the stop is inferred from content). Also `SupportsUsageInStreaming` (nil = send `stream_options.include_usage` so token usage streams; set false for servers that reject `stream_options`; reports are parsed whenever present regardless). The caller sets the `Compat` matching their model when it deviates from the OpenAI default.
_Avoid_: ModelCompat, Quirks

**Provider stop**: A provider-terminated message: the stream ended with a terminal stop/finish reason that has no portable mapping (Gemini SAFETY/RECITATION/..., OpenAI `content_filter`/`network_error`, or a genuinely unknown reason). Fatal: surfaces as a non-transient `LLMErrorEvent` wrapping `ErrProviderStop` — never as a successful `StopReasonUnknown` (upstream #7272). Error reasons win over toolUse even with calls in flight: an error stop mid-call means arguments may be incomplete, the same rationale as the length-wins rule (upstream #6285). Distinct from a missing finish reason, which wraps `ErrMalformedResponse` unless `Compat.SupportsFinishReason` says the provider omits it.
_Avoid_: error stop, finish error

**ThinkingFormat**: The wire mechanism an OpenAI-compatible model uses to toggle reasoning, selected by `Compat.ThinkingFormat`: `reasoning_effort` (OpenAI o-series + xAI/Mistral — the default; gated per model by the compat model class), `deepseek` (`thinking:{type:enabled/disabled}` plus optional `reasoning_effort`; DeepSeek V4 on the opencode-go gateway, reused by z.ai/GLM), `chat-template` (`chat_template_kwargs.enable_thinking`, for Qwen/DeepSeek behind vLLM), and `qwen` (top-level `enable_thinking` bool, for Alibaba DashScope). Adding a new format is a new enum value plus one branch, never a catalog.
_Avoid_: ReasoningMode, ThinkingMechanism

### Retry

**RetryPolicy**: Shared config shape on every provider `Config` (`Retry` field): `MaxRetries`, `MaxRetryDelay`. Consumed by the provider **retry ladder** (`llm.Retry`, LLM-18), not delegated to an SDK. The zero value resolves to the documented defaults (3 retries, 60s cap — upstream's agent policy); negative fields disable (`MaxRetries: -1` runs once; `MaxRetryDelay: -1` lifts the cap on server-requested waits). Delay rules (upstream `getRetryDelayMs`): a `retry-after`/`retry-after-ms` server hint wins and **fails immediately when above the cap**; otherwise exponential `min(0.5·2^attempt, 8)s` with −0–25% jitter.

**Retry ladder**: The single shared implementation in `llm.Retry`; each provider wraps its **stream-open phase** in it, never mid-stream, so content is never duplicated. Providers own classification: transient open failures (transport errors — including DNS, upstream #6946 — 408/409/429, 5xx; `x-should-retry` header override on openai-compat) are returned as `*llm.TransientError{Err, RetryAfter}` and retried per `RetryPolicy`; `ErrProviderFatal`/`ErrContextOverflow`/`ErrProviderStop` and other 4xx pass through unretried. Exhausted retries return the last `TransientError`, surfacing as `LLMErrorEvent{Transient: true}` + `StreamResult.Err` exactly as before. The genai SDK exposes no response headers, so the Gemini side retries with backoff only (no hints).

**LLMRetryEvent**: Emitted by the ladder before each retry wait — `Attempt` (1-based), `NextDelay`, `Reason`, `ServerHint` (true when a `retry-after` header drove the delay), `Provider`, `Model`.

**TransientError**: The wrapper a provider's retry-ladder op returns for retryable stream-open failures. `RetryAfter` carries the server's requested wait (0 = no hint → backoff). `Unwrap`s to the classified error, so `errors.Is/As` chains survive the ladder.

### Testing

**MockProvider**: Fluent-builder `LLMProvider` in `mock/` subpackage. Scripts responses keyed by matcher.
