package gemini

import (
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
	"google.golang.org/genai"
)

// The portable totals for one request: input = prompt − cached content;
// output = candidates + thoughts (upstream google-generative-ai.ts formula).
func TestUsageEventFromMetadata(t *testing.T) {
	for _, tt := range []struct {
		name string
		meta *genai.GenerateContentResponseUsageMetadata
		want llm.UsageEvent
	}{
		{"plain", &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 100, CandidatesTokenCount: 40,
		}, llm.UsageEvent{InputTokens: 100, OutputTokens: 40}},
		{"cached subtracted", &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 100, CachedContentTokenCount: 30, CandidatesTokenCount: 40,
		}, llm.UsageEvent{InputTokens: 70, OutputTokens: 40}},
		{"thoughts counted in output", &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount: 100, CandidatesTokenCount: 40, ThoughtsTokenCount: 60,
		}, llm.UsageEvent{InputTokens: 100, OutputTokens: 100}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := usageEventFromMetadata(tt.meta); got != tt.want {
				t.Errorf("usageEventFromMetadata(%+v) = %+v, want %+v", tt.meta, got, tt.want)
			}
		})
	}
}
