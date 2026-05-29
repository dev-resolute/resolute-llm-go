package llm

// ThinkingLevel is a portable abstraction over provider-specific reasoning controls.
type ThinkingLevel int

const (
	ThinkingOff ThinkingLevel = iota
	ThinkingMinimal
	ThinkingLow
	ThinkingMedium
	ThinkingHigh
)

// ProviderHints is a typed escape hatch for provider-specific config.
// Only the field matching the active provider is consulted; others are ignored.
type ProviderHints struct {
	OpenAI *OpenAIHints
	Gemini *GeminiHints
}

// OpenAIHints carries provider-specific overrides for the OpenAI-compatible adapter.
type OpenAIHints struct {
	ReasoningEffort string
}

// GeminiHints carries provider-specific overrides for the Gemini provider.
type GeminiHints struct {
	ThinkingBudget int
}
