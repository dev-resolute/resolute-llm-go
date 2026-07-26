package gemini

import (
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
	"google.golang.org/genai"
)

func TestMapGeminiFinishReason(t *testing.T) {
	for _, tt := range []struct {
		reason       genai.FinishReason
		sawToolCalls bool
		want         llm.StopReason
	}{
		{genai.FinishReasonMaxTokens, true, llm.StopReasonLength},
		{genai.FinishReasonMaxTokens, false, llm.StopReasonLength},
		{genai.FinishReasonStop, true, llm.StopReasonToolUse},
		{genai.FinishReasonStop, false, llm.StopReasonStop},
		{genai.FinishReason(""), false, llm.StopReasonUnknown},
	} {
		if got := mapGeminiFinishReason(tt.reason, tt.sawToolCalls); got != tt.want {
			t.Errorf("mapGeminiFinishReason(%q, %v) = %q, want %q", tt.reason, tt.sawToolCalls, got, tt.want)
		}
	}
}
