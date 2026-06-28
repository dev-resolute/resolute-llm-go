//go:build integration

package openaicompat

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

// TestLiveZAIGLMThinking exercises GLM end-to-end through the ZAI constructor: the
// thinking:{type} dialect on glm-4.6 and the reasoning_content round-trip. Set
// ZAI_API_KEY to run.
func TestLiveZAIGLMThinking(t *testing.T) {
	key := os.Getenv("ZAI_API_KEY")
	if key == "" {
		t.Skip("ZAI_API_KEY not set")
	}

	p, err := ZAI(TargetConfig{APIKey: key})
	if err != nil {
		t.Fatalf("ZAI: %v", err)
	}

	req := llm.LLMRequest{
		Model:    "glm-4.6",
		Thinking: llm.ThinkingHigh,
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "What is 17 * 23? Reason step by step, then state the final number."}},
		},
	}
	stream := p.Stream(context.Background(), req)
	var thinking, answer strings.Builder
	for ev := range stream.Events {
		switch e := ev.(type) {
		case llm.ThinkingDeltaEvent:
			thinking.WriteString(e.Delta)
		case llm.TextDeltaEvent:
			answer.WriteString(e.Delta)
		}
	}
	if res := <-stream.Done; res.Err != nil {
		t.Fatalf("stream: %v", res.Err)
	}

	if thinking.Len() == 0 {
		t.Error("expected reasoning content from glm-4.6 (thinking enabled), got none")
	}
	if !strings.Contains(answer.String(), "391") {
		t.Errorf("answer missing the correct product 391; got %q", answer.String())
	}
}
