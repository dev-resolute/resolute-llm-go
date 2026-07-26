package gemini

import (
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

func TestSessionIDIgnored(t *testing.T) {
	// given two requests identical except for SessionID
	base := llm.LLMRequest{Model: "gemini-2.5-flash", Thinking: llm.ThinkingHigh}
	withSession := base
	withSession.SessionID = "sess-abc"

	// when the provider builds the genai config for each
	got, err := toGeminiConfig(withSession, nil)
	if err != nil {
		t.Fatalf("toGeminiConfig(withSession): unexpected error: %v", err)
	}
	want, err := toGeminiConfig(base, nil)
	if err != nil {
		t.Fatalf("toGeminiConfig(base): unexpected error: %v", err)
	}

	// then SessionID does not leak into Labels or CachedContent
	if len(got.Labels) != 0 {
		t.Errorf("Labels = %v, want empty (SessionID must not leak)", got.Labels)
	}
	if got.CachedContent != want.CachedContent {
		t.Errorf("CachedContent = %q, want %q (SessionID must not affect it)", got.CachedContent, want.CachedContent)
	}
}
