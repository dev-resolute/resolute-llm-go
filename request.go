package llm

// LLMRequest carries all inputs for a single streaming LLM call.
type LLMRequest struct {
	Model         string
	Messages      []Message
	Tools         []ToolDef
	Thinking      ThinkingLevel
	ProviderHints ProviderHints
	Retry         RetryPolicy
}
