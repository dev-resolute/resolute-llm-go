package openaicompat

import (
	"regexp"
	"strings"

	"github.com/dev-resolute/resolute-llm-go"
)

// xaiBaseURL is xAI's OpenAI-compatible endpoint.
const xaiBaseURL = "https://api.x.ai/v1"

// XAI builds a provider for xAI's Grok models against the OpenAI-compatible
// endpoint. Reasoning is toggled with reasoning_effort, gated per model
// (grok-4 and the -fast variants always reason and reject the param).
func XAI(cfg TargetConfig) (llm.LLMProvider, error) {
	return newTarget(cfg, "xai", xaiBaseURL, Compat{}, classifyXAI)
}

var (
	reXAIGrok4      = regexp.MustCompile(`^grok-4`)
	reXAIGrok4Minor = regexp.MustCompile(`^grok-4\.\d`)
)

// classifyXAI maps an xAI model id to its capabilities. Grok-4 and the -fast
// variants always reason but reject reasoning_effort (HTTP 400); grok-3-mini and
// the dotted-minor grok-4.x line accept it; plain grok-3 does not reason.
func classifyXAI(model string) classification {
	m := strings.ToLower(model)
	grok4 := reXAIGrok4.MatchString(m)
	mini := strings.Contains(m, "grok-3-mini")
	return classification{
		thinking:        grok4 || mini,
		vision:          grok4 || strings.Contains(m, "vision"),
		reasoningEffort: mini || reXAIGrok4Minor.MatchString(m),
		strictTools:     true,
	}
}
