package llm

// ProviderCapabilities describes the feature set of a specific model.
type ProviderCapabilities struct {
	Streaming         bool
	ToolCalling       bool
	ParallelToolCalls bool
	Thinking          bool
	PromptCaching     bool
	Vision            bool
}
