package openaicompat

import (
	"regexp"
	"strings"

	"github.com/dev-resolute/resolute-llm-go"
)

// zaiBaseURL is z.ai's OpenAI-compatible endpoint for Zhipu GLM models.
const zaiBaseURL = "https://api.z.ai/api/paas/v4"

// ZAI builds a provider for Zhipu's GLM models against the z.ai endpoint. GLM-4.5
// and GLM-4.6 share DeepSeek's thinking:{type:enabled|disabled} wire shape, so the
// provider uses the ThinkingDeepSeek dialect; GLM does not require echoing
// reasoning_content back on assistant messages.
func ZAI(cfg TargetConfig) (llm.LLMProvider, error) {
	return newTarget(cfg, "zai", zaiBaseURL, Compat{ThinkingFormat: ThinkingDeepSeek}, classifyZAI)
}

// reZAIVision matches the GLM vision models, whose ids carry a "v" right after the
// version number (glm-4.5v, glm-4v, glm-4v-plus).
var reZAIVision = regexp.MustCompile(`glm-\d(\.\d)?v`)

// classifyZAI maps a GLM model id to its capabilities. The GLM-4.5 and GLM-4.6
// lines are hybrid reasoning models; the "v" vision variants accept image input.
func classifyZAI(model string) classification {
	m := strings.ToLower(model)
	return classification{
		thinking:    strings.HasPrefix(m, "glm-4.5") || strings.HasPrefix(m, "glm-4.6"),
		vision:      reZAIVision.MatchString(m),
		strictTools: true,
	}
}
