//go:build integration

package openaicompat

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

// TestLiveOpenCodeGoDeepSeekThinking exercises DeepSeek V4 on the opencode-go
// gateway end-to-end: the deepseek thinking format, reasoning_content round-trip,
// and max_tokens. Set OPENCODE_API_KEY to run.
func TestLiveOpenCodeGoDeepSeekThinking(t *testing.T) {
	key := os.Getenv("OPENCODE_API_KEY")
	if key == "" {
		t.Skip("OPENCODE_API_KEY not set")
	}

	p, err := New(Config{
		BaseURL: "https://opencode.ai/zen/go/v1",
		APIKey:  key,
		Compat: Compat{
			ThinkingFormat: ThinkingDeepSeek,
			RequiresReasoningContentOnAssistantMessages: true,
			MaxTokens: 8000,
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := llm.LLMRequest{
		Model:    "deepseek-v4-pro",
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
		t.Error("expected reasoning content from DeepSeek (thinking enabled), got none")
	}
	if !strings.Contains(answer.String(), "391") {
		t.Errorf("answer missing the correct product 391; got %q", answer.String())
	}
}
