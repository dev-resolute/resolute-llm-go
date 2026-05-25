# pi-llm-go — Glossary

## LLM Layer Terms

### Provider model

**LLMProvider**: The single interface every concrete provider implements. Exposes streaming, tool calls, and thinking blocks in a normalized event shape.

**OpenAI-compatible adapter**: One `LLMProvider` implementation, parameterized by base URL, covering Fireworks and locally-hosted servers (Ollama, vLLM, llama.cpp server, LM Studio).

**Gemini provider**: The `LLMProvider` implementation built on `google.golang.org/genai`.

### Streaming

**EventStream**: Struct with `Events <-chan LLMEvent` and `Done <-chan StreamResult`. The shared return shape for any LLM streaming call.

**StreamResult**: Terminal value on `EventStream.Done`. Contains final messages + error.

**LLMEvent**: Sealed interface for events on `EventStream.Events`. Concrete variants: `TextDeltaEvent`, `ThinkingDeltaEvent`, `ToolCallStartEvent`, `ToolCallEndEvent`, `MessageEndEvent`, `LLMErrorEvent`, `LLMRetryEvent`, `UsageEvent`.

### Messages

**Message**: LLM-side unit of content. Struct with `Role` and sealed `Content` interface. Distinct from agent-side transcript `Message`.

**Content**: Sealed interface with variants `TextContent`, `ToolCallContent`, `ToolResultContent`, `ThinkingContent`.

**ToolDef**: LLM-visible tool spec. `Name`, `Description`, `Schema json.RawMessage`.

### Thinking and capabilities

**ThinkingLevel**: Portable enum (`ThinkingOff`, `ThinkingLow`, `ThinkingMedium`, `ThinkingHigh`).

**ProviderHints**: Typed escape hatch with nilable typed pointers (`*OpenAIHints`, `*GeminiHints`).

**ProviderCapabilities**: Concrete bool fields per model: `Streaming`, `ToolCalling`, `ParallelToolCalls`, `Thinking`, `PromptCaching`, `Vision`.

### Retry

**RetryPolicy**: Shared config shape. `MaxRetries`, `MaxRetryDelay`. Delegated to underlying SDK.

**LLMRetryEvent**: Emitted per retry attempt for observability.

### Testing

**MockProvider**: Fluent-builder `LLMProvider` in `mock/` subpackage. Scripts responses keyed by matcher.
