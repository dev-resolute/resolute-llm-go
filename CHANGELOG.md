# Changelog

## [0.12.0] - 2026-08-09

> **Live validation:** full `gemini` integration suite passes against the live API —
> tool-call thought-signature round trip incl. `FunctionCall.ID` replay and the strict
> VALIDATED round trip on `gemini-3.1-pro-preview`, `TestLiveGeminiUsageEvent` on
> `gemini-2.5-flash` — plus agent-core's `TestLiveGemini3AgentToolLoop`. Ports upstream
> pi 0.82.1–0.84.1 (rediff record: `docs/issues/REDIFF-0.82.0-to-0.84.1.md`).

### Breaking

- **Unmapped and missing finish reasons are now fatal errors, not silent
  `StopReasonUnknown` stops (LLM-17, upstream #7272 parity).** Both providers previously
  mapped any unrecognized terminal reason — and any stream that ended without a finish
  reason — to `StopReasonUnknown`, a silent "success". Now: Gemini
  `SAFETY`/`RECITATION`/`LANGUAGE`/`OTHER`/`BLOCKLIST`/`PROHIBITED_CONTENT`/`SPII`/
  `MALFORMED_FUNCTION_CALL`/`IMAGE_SAFETY`/`UNEXPECTED_TOOL_CALL`/
  `IMAGE_PROHIBITED_CONTENT`/`NO_IMAGE`/`IMAGE_RECITATION`/`IMAGE_OTHER`/
  `FINISH_REASON_UNSPECIFIED`, OpenAI `content_filter`/`network_error`, and genuinely
  unknown reasons surface as a **non-transient `LLMErrorEvent` wrapping the new
  `llm.ErrProviderStop`** and a non-nil `StreamResult.Err` (no `MessageEndEvent`).
  A stream ending without any finish reason is a protocol error wrapping
  `llm.ErrMalformedResponse` (upstream's `"pending"` invariant), unless the provider
  opts out via `Compat.SupportsFinishReason`. **Error reasons win over toolUse even
  with calls in flight** (deliberate divergence: upstream maps any reason to toolUse
  when a toolCall block exists, silently bypassing its own #6285 truncation guard).
  `StopReasonUnknown` remains in the enum as the zero value but is no longer emitted
  by either shipped provider on a successful stream.
- **Providers now retry transient stream-open failures — default-on (LLM-18, upstream
  `retryProviderRequest` port).** `RetryPolicy` is finally consumed by both providers,
  and `LLMRetryEvent` is finally emitted. The zero value resolves to the documented
  defaults (**3 retries, 60s cap** — upstream's agent policy), so streams that used to
  fail immediately on a 429/5xx/transport blip (including DNS failures, upstream #6946)
  now succeed after a wait; **opt out with `MaxRetries: -1`**, and lift the cap on
  server-requested waits with `MaxRetryDelay: -1`. The retried boundary is the
  **stream-open phase only** — streaming proper is never retried, so content is never
  duplicated. Rules match upstream `getRetryDelayMs`: `retry-after`/`retry-after-ms`
  hints win and **fail immediately above the cap** (message names both delays);
  otherwise exponential `min(0.5·2^attempt, 8)s` with −0–25% jitter;
  `x-should-retry: true|false` response header overrides classification
  (openai-compat). Exhausted retries surface exactly as before
  (`LLMErrorEvent{Transient: true}` + `StreamResult.Err`), and intermediate attempts
  now emit `LLMRetryEvent` instead of per-attempt `LLMErrorEvent`s.

### Added

- **Text/thinking thought signatures round-trip (LLM-16, upstream #7362).**
  `ThoughtSignature` on `TextContent`/`ThinkingContent` and on
  `TextDeltaEvent`/`ThinkingDeltaEvent`: Gemini attaches signatures to text and
  thinking parts — including parts whose visible text is empty — and requires them
  echoed back on replay, or the reasoning chain breaks. The provider emits the
  signature on delta events (typically one delta of a part, possibly with an empty
  `Delta`; consumers retain the last non-empty value) and `toGeminiContents` replays
  it verbatim. Empty assistant text/thinking blocks are dropped only when unsigned;
  user text is never filtered; `ThinkingContent` now converts to a real
  `Thought: true` part instead of flattened text.
- **Tool-call IDs on Gemini 3 history conversion (LLM-16, upstream #7494).**
  `toGeminiContents` sets `FunctionCall.ID`/`FunctionResponse.ID` from the recorded
  `CallID` when the model class is Gemini 3 (new `requiresToolCallID` gate) — signed
  multi-turn replay breaks without them. Pre-Gemini-3 models and Gemma 4 get no IDs.
- **`Compat.SupportsFinishReason *bool` (LLM-17, upstream 0.84.0).** nil = the
  provider is expected to terminate streams with `finish_reason`; set `false` for
  providers known to omit it (some vLLM/llama.cpp-style local servers) — the stop is
  then inferred from content, and tool calls still buffered at stream end are flushed
  with finalized arguments instead of being dropped.
- **Providers emit `UsageEvent` (LLM-19) — no longer a dead type.** At most one per
  stream, final totals, immediately before `MessageEndEvent` (last report wins), so
  consumer-side accumulation (Compact's `BranchSummary.Usage`) is exactly-once per
  call. openai-compat sends `stream_options: {include_usage: true}` (new
  `Compat.SupportsUsageInStreaming *bool`, nil = send) and parses chunk-level `usage`
  plus the Moonshot-style per-choice fallback (`input = max(0, prompt − cached −
  cache_write)`; llama.cpp reports usage once asked — upstream #7258 by construction);
  gemini maps `usageMetadata` (`input = prompt − cached`; `output = candidates +
  thoughts`).
- **`llm.Retry` and `llm.TransientError` — the shared retry ladder (LLM-18).**
  Providers wrap their stream-open phase in `llm.Retry(ctx, policy, provider, model,
  emit, op)`; classification stays provider-owned via `*TransientError{Err,
  RetryAfter}` returns. openai-compat re-resolves `GetAPIKey` per attempt (expiring
  credentials refresh across retries); gemini retries with backoff only (the genai
  SDK exposes no response headers).

## [0.11.0] - 2026-07-26

> **Live validation:** `TestLiveGeminiStrictPreferToolRoundTrip` passes against the live
> `gemini-3.1-pro-preview` API.

### Added

- **`ToolDef.ConstrainedSampling` — strict JSON-schema tool sampling (LLM-12, upstream
  0.82.0 `constrained-sampling.ts`).** A tool opts in with
  `ConstrainedSampling: &llm.ConstrainedSampling{Strict: llm.StrictPrefer}` or
  `llm.StrictRequire`; nil (the default) is unchanged, off behaviour. The new
  `llm.ResolveStrictSampling(tool, supported)` is the single resolver both adapters call:
  `prefer` uses strict sampling when the provider/model supports it and **silently falls
  back** (no error, no field) when it doesn't; `require` **fails fatally** when unsupported,
  with the exact message `Tool "<name>" requires JSON-schema constrained sampling, but
  strict tools are unsupported.`, wrapped in `llm.ErrProviderFatal` so retry ladders settle
  on the first attempt instead of retrying a config error that will never succeed (LLM-11
  polarity). A non-nil `ConstrainedSampling` whose `Strict` is anything other than
  `prefer`/`require` (including the zero value `""`) is also a loud resolver error, naming
  the offending tool.
- **openai-compat strict tool-call emission, per-instance gated (LLM-12).** New
  `Config.SupportsStrictTools *bool`: nil defers to the classifier default, which is
  **true** for a plain instance (no classifier) and every current named family (xAI,
  Mistral, Qwen, z.ai) — mirroring upstream, where only moonshot/together/cloudflare-gateway/
  nvidia are denylisted, none of which are shipped families today; a non-nil pointer
  (including one pointing at `false`) always overrides the classifier, in either direction.
  When the resolved instance supports strict, **every** function tool on the wire carries a
  `"strict"` key — `true` or `false` per `ResolveStrictSampling`'s per-tool result, even for
  tools that never opted in. When it doesn't, the `"strict"` key is **omitted entirely**
  from every tool (some providers reject unrecognized fields with HTTP 400). A `require`
  tool on an unsupported instance fails before any HTTP request is issued, as a fatal
  `LLMErrorEvent`/`StreamResult.Err`.
- **Gemini VALIDATED function-calling mode for strict tools on Gemini 3 (LLM-12).** Support
  is version-derived, not configured: `classifyGemini(model) ∈ {class3Pro, class3Flash}` →
  supported; Gemini 2.5, legacy Gemini, and **Gemma 4** (upstream's `^gemini-` gate never
  matches Gemma ids) → unsupported. When any tool on the request resolves strict, the
  provider sets `ToolConfig.FunctionCallingConfig.Mode = VALIDATED` for the **whole request**
  (upstream `google-shared.ts:311-324` — the mode is request-level, not per-tool); with no
  strict-resolved tool, `ToolConfig` stays nil. `prefer` on a model below Gemini 3 falls back
  silently (no `ToolConfig`, no error); `require` on a model below Gemini 3 fails fatally
  before the request is sent, via the same `ResolveStrictSampling` + `llm.ErrProviderFatal`
  path as openai-compat.

**Grammar-constrained custom tools (`openai_lark`/`openai_regex`, `type:"custom"` tools,
streaming input re-encoding) are deliberately deferred**, not ported in this release.
Upstream only ever enables grammar sampling on Responses-API endpoints (GPT-5+ on
openai/codex/azure/copilot/opencode/cloudflare); no chat-completions model gets it, and
today's adapters here are Gemini and openai-compat chat-completions — there is no target
that could use it yet. Tracked as a watch item, triggered by a future Responses-family
adapter or a named compat instance verified to accept OpenAI custom tools.

## [0.10.1] - 2026-07-26

### Fixed

- **openai-compat tool-call flush order is now deterministic.** `readSSE` flushed buffered tool
  calls by ranging over the `toolCallBufs` map, so for an assistant message with multiple tool
  calls, the emitted `ToolCallEndEvent` order — and `StreamResult.Messages` order — was
  nondeterministic run-to-run (Go randomizes map iteration order). A new `toolCallOrder` slice
  records first-appearance order as each call ID is buffered, and the flush now iterates that
  slice (looking up the map) instead of ranging the map directly, so calls are always emitted in
  the order the model streamed them — matching upstream content order and the execution order
  agent-core now derives from these events.

## [0.10.0] - 2026-07-25

> **Live validation:** `TestLiveGeminiThinkingSurfaces_2_5_Flash`,
> `TestLiveGemini3ToolCallThoughtSignatureRoundTrip`, and
> `TestLiveGeminiThinkingSurfaces_3_Flash` pass against the live Gemini API.

### Added

- **`ThinkingXhigh` and `ThinkingMax` levels (LLM-13, upstream 0.80.6+).** Two new
  `llm.ThinkingLevel` values above `ThinkingHigh`. The openai-compat adapter passes them straight
  through as `reasoning_effort: "xhigh"` / `"max"` (models that don't support the extra levels
  simply see an effort string they ignore or reject same as any unsupported value today). Gemini
  has no levels above `HIGH`, so both clamp to `genai.ThinkingLevelHigh` with the same 16000-token
  default budget as `ThinkingHigh` (still overridable via `req.ThinkingBudgets`).
- **`StopReason` on `MessageEndEvent` (LLM-15).** New `llm.StopReason` string type —
  `StopReasonStop`, `StopReasonLength`, `StopReasonToolUse`, `StopReasonUnknown` — mirrors
  upstream's stop-reason set (minus error/aborted, which this API already signals via
  `LLMErrorEvent`/`StreamResult.Err`). Both providers map their native finish reason onto it, with
  a length-truncation callout: `length` (openai-compat) / `MAX_TOKENS` (Gemini) wins over tool use,
  since a truncated message's tool-call arguments may be incomplete (upstream #6285) and callers
  need to be able to tell.
- **`ToolCallEndEvent` carries the finalized call (LLM-15).** `ToolName`, `Args`
  (`json.RawMessage`), and `ThoughtSignature` are new fields on `ToolCallEndEvent` — for providers
  that stream arguments incrementally, this event (not `ToolCallStartEvent`) is where complete,
  ready-to-execute arguments appear. Both providers now populate all three on flush.

### Fixed

- **openai-compat streamed tool-call arguments now reach event consumers (LLM-15).** The SSE
  reader only flushed accumulated tool-call buffers into `ToolCallEndEvent`/`StreamResult.Messages`
  when `finish_reason == "tool_calls"`; any other terminal reason — notably `"length"` on a
  truncated response — dropped the buffered arguments entirely, so event-driven callers received
  no tool call at all (only `StreamResult.Messages`-based callers were unaffected, and even those
  had no way to know the call was truncated). Flush now happens on any non-empty `finish_reason`,
  and the emitted `ToolCallEndEvent` carries the finalized `ToolName`/`Args` so a length-truncated
  call is both delivered and identifiable via the new `StopReason`.
- **Gemini `functionResponse` wire format changes to `{"output"}`/`{"error"}` (upstream
  `google-shared.ts` parity — WIRE FORMAT CHANGE).** The response body's key changes from
  `{"result": ...}` to `{"output": ...}` on success or `{"error": ...}` when
  `ToolResultContent.IsError` is set (previously `IsError` was silently dropped and everything
  went under `result`). Any inspection or replay logic keyed on the old `result` field must move
  to `output`/`error`.
- **Consecutive Gemini tool results merge into one turn.** Consecutive tool-result messages are
  now sent as a single `user` turn carrying multiple `functionResponse` parts, instead of one turn
  per result, matching upstream's Cloud Code Assist-compatible layout.
- **Gemini ≥ 3 / Gemma 4 nest tool-result images in `functionResponse.parts`.** Per upstream's
  major-version gate, tool-result images are now nested inside the `functionResponse` part for
  Gemini 3+ and Gemma 4 models. Gemini 2.5 and legacy models are unaffected and keep the separate
  trailing `"Tool result image:"` user turn.

## [0.9.0] - 2026-07-25

### Added

- **`ImageContent` (AGENT-18 R1, upstream 0.82.0 read-tool support).** New sealed
  content type carrying raw image bytes + MIME type; valid as a user message
  (a text+image turn is two adjacent user messages) and on the new additive
  `ToolResultContent.Images`. Gemini sends `inlineData` parts, with tool-result
  images as a trailing `"Tool result image:"` user turn (upstream's
  universally-supported google-shared layout; the Gemini-3 nested multimodal
  functionResponse is a recorded follow-up). The OpenAI-compatible adapter sends
  `image_url` data URLs and batches images from consecutive tool results into one
  trailing user message led by an `"Attached image(s) from tool result:"` text
  part (upstream parity); empty tool-result text becomes `"(see attached image)"`
  when images are attached, else `"(no tool output)"` (upstream three-way
  placeholder parity). No capability gating — per ADR-0008 apps own model choice.

## [0.8.2] - 2026-07-21

### Fixed

- **OpenAI-compatible conversion keeps tool call IDs unique on cross-provider
  replay (port of upstream 0.81.0, pi#6854).** `toOpenAIMessages` previously
  replayed `ToolCallContent.CallID` verbatim as both `tool_calls[].id` and the
  matching `tool_call_id`. Transcripts crossing providers — e.g. Gemini
  function calls, whose IDs are empty — produced duplicate or empty IDs on the
  wire, breaking call/result pairing. Conversion now assigns `call_N` to empty
  IDs and suffixes duplicates (`id_2`, `id_3`, …, capped at OpenAI's 40-char
  limit), remapping tool results to their call in occurrence order. Unique
  non-empty IDs pass through unchanged.

## [0.8.1] - 2026-07-04

### Fixed

- **Deterministic Gemini 4xx client errors now classify as `ErrProviderFatal` (LLM-11).** The
  gemini adapter wraps stream errors carrying `genai.APIError` HTTP 400/401/403/404 or status
  `INVALID_ARGUMENT`/`FAILED_PRECONDITION` in `llm.ErrProviderFatal`, so retry ladders
  (resolute-harness-go `runRecovered`) settle such requests on the first attempt instead of
  burning an attempt budget on a request the provider rejects identically every time. 429
  (`RESOURCE_EXHAUSTED`), 5xx, and transport errors stay retryable, and context-overflow 400s
  pass through unwrapped so LLM-8 compact-and-retry handling keeps working. The underlying
  `genai.APIError` remains reachable via `errors.As`.

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
