//go:build integration

package openaicompat

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

// TestLiveXAIGrokReasoning exercises Grok end-to-end through the XAI constructor:
// the reasoning_effort dialect on grok-3-mini (which accepts the param) plus the
// reasoning_content round-trip. Set XAI_API_KEY to run.
func TestLiveXAIGrokReasoning(t *testing.T) {
	key := os.Getenv("XAI_API_KEY")
	if key == "" {
		t.Skip("XAI_API_KEY not set")
	}

	p, err := XAI(TargetConfig{APIKey: key})
	if err != nil {
		t.Fatalf("XAI: %v", err)
	}

	req := llm.LLMRequest{
		Model:    "grok-3-mini",
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
		t.Error("expected reasoning content from grok-3-mini (thinking enabled), got none")
	}
	if !strings.Contains(answer.String(), "391") {
		t.Errorf("answer missing the correct product 391; got %q", answer.String())
	}
}
