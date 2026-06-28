//go:build integration

package openaicompat

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

// TestLiveQwenThinking exercises Qwen end-to-end through the Qwen constructor: the
// top-level enable_thinking dialect on qwen-plus and the reasoning_content
// round-trip (DashScope requires streaming when thinking is on, which the provider
// always does). Set DASHSCOPE_API_KEY to run.
func TestLiveQwenThinking(t *testing.T) {
	key := os.Getenv("DASHSCOPE_API_KEY")
	if key == "" {
		t.Skip("DASHSCOPE_API_KEY not set")
	}

	p, err := Qwen(TargetConfig{APIKey: key})
	if err != nil {
		t.Fatalf("Qwen: %v", err)
	}

	req := llm.LLMRequest{
		Model:    "qwen-plus",
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
		t.Error("expected reasoning content from qwen-plus (thinking enabled), got none")
	}
	if !strings.Contains(answer.String(), "391") {
		t.Errorf("answer missing the correct product 391; got %q", answer.String())
	}
}
