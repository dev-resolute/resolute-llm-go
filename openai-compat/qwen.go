package openaicompat

import (
	"strings"

	"github.com/dev-resolute/resolute-llm-go"
)

// qwenBaseURL is Alibaba DashScope's international OpenAI-compatible endpoint.
const qwenBaseURL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

// Qwen builds a provider for Alibaba's Qwen models against the DashScope
// compatible-mode endpoint. Reasoning is toggled with a top-level enable_thinking
// bool; the provider always streams, satisfying DashScope's constraint that
// commercial Qwen3 models must stream when thinking is on.
func Qwen(cfg TargetConfig) (llm.LLMProvider, error) {
	return newTarget(cfg, "qwen", qwenBaseURL, Compat{ThinkingFormat: ThinkingQwen}, classifyQwen)
}

// classifyQwen maps a Qwen model id to its capabilities. The Qwen3 chat models,
// the commercial qwen-max/plus/flash/turbo line, and the QwQ models reason
// (excluding the coder models); the -vl line accepts image input.
func classifyQwen(model string) classification {
	m := strings.ToLower(model)
	commercial := strings.HasPrefix(m, "qwen-max") || strings.HasPrefix(m, "qwen-plus") ||
		strings.HasPrefix(m, "qwen-flash") || strings.HasPrefix(m, "qwen-turbo")
	reasons := strings.HasPrefix(m, "qwen3") || strings.HasPrefix(m, "qwq") || commercial
	return classification{
		thinking: reasons && !strings.Contains(m, "coder"),
		vision:   strings.Contains(m, "-vl"),
	}
}
