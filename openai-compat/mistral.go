package openaicompat

import (
	"strings"

	"github.com/dev-resolute/resolute-llm-go"
)

// mistralBaseURL is Mistral's OpenAI-compatible endpoint.
const mistralBaseURL = "https://api.mistral.ai/v1"

// Mistral builds a provider for Mistral models against the OpenAI-compatible
// endpoint. The hybrid mistral-small/medium models toggle reasoning with
// reasoning_effort; the Magistral models reason natively and omit the param.
func Mistral(cfg TargetConfig) (llm.LLMProvider, error) {
	return newTarget(cfg, "mistral", mistralBaseURL, Compat{}, classifyMistral)
}

// classifyMistral maps a Mistral model id to its capabilities. The hybrid
// mistral-small/medium line reasons via reasoning_effort; the Magistral line
// reasons natively (no effort param); pixtral and the multimodal small/medium
// accept image input.
func classifyMistral(model string) classification {
	m := strings.ToLower(model)
	hybrid := strings.HasPrefix(m, "mistral-small") || strings.HasPrefix(m, "mistral-medium")
	magistral := strings.HasPrefix(m, "magistral")
	return classification{
		thinking:        hybrid || magistral,
		vision:          hybrid || strings.HasPrefix(m, "magistral-medium") || strings.HasPrefix(m, "pixtral"),
		reasoningEffort: hybrid,
		strictTools:     true,
	}
}
