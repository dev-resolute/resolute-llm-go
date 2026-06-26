package openaicompat

import "github.com/resolute-sh/pi-llm-go"

// ThinkingFormat selects how an OpenAI-compatible model toggles reasoning on the
// wire. Adding a format is a new value plus one branch in toRequestBody (ADR-0008).
type ThinkingFormat int

const (
	// ThinkingReasoningEffort is the OpenAI default: a reasoning_effort field.
	ThinkingReasoningEffort ThinkingFormat = iota
	// ThinkingDeepSeek toggles reasoning via thinking:{type:enabled|disabled},
	// optionally alongside reasoning_effort. Used by DeepSeek V4 on opencode-go.
	ThinkingDeepSeek
	// ThinkingChatTemplate toggles reasoning via chat_template_kwargs.enable_thinking.
	// Used by local reasoning models (Qwen3, DeepSeek-R1) served behind vLLM/llama.cpp.
	ThinkingChatTemplate
)

// Compat carries per-model behaviour the caller supplies because pi-llm-go ships
// no model catalog (ADR-0008). It mirrors the upstream compat object and is set on
// Config to match the model a provider instance targets.
type Compat struct {
	// ThinkingFormat selects the wire mechanism for toggling reasoning.
	ThinkingFormat ThinkingFormat
	// SupportsReasoningEffort additionally sends reasoning_effort when thinking is
	// on under a non-default ThinkingFormat.
	SupportsReasoningEffort bool
	// RequiresReasoningContentOnAssistantMessages ensures every assistant message
	// carries a reasoning_content field; DeepSeek rejects messages without it.
	RequiresReasoningContentOnAssistantMessages bool
	// MaxTokens, when greater than zero, is sent as the max_tokens body field. Some
	// gateways (opencode-go) require it.
	MaxTokens int
}

// reasoningEffort maps the portable thinking level to an OpenAI reasoning_effort
// string, honouring a per-call ProviderHints override.
func reasoningEffort(req llm.LLMRequest) string {
	effort := map[llm.ThinkingLevel]string{
		llm.ThinkingMinimal: "minimal",
		llm.ThinkingLow:     "low",
		llm.ThinkingMedium:  "medium",
		llm.ThinkingHigh:    "high",
	}[req.Thinking]
	if req.ProviderHints.OpenAI != nil && req.ProviderHints.OpenAI.ReasoningEffort != "" {
		effort = req.ProviderHints.OpenAI.ReasoningEffort
	}
	return effort
}
