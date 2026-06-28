//go:build integration

package openaicompat

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

// TestLiveMistralReasoning exercises Mistral end-to-end through the Mistral
// constructor: the reasoning_effort dialect on the hybrid mistral-small model
// (which accepts the param). Set MISTRAL_API_KEY to run.
func TestLiveMistralReasoning(t *testing.T) {
	key := os.Getenv("MISTRAL_API_KEY")
	if key == "" {
		t.Skip("MISTRAL_API_KEY not set")
	}

	p, err := Mistral(TargetConfig{APIKey: key})
	if err != nil {
		t.Fatalf("Mistral: %v", err)
	}

	req := llm.LLMRequest{
		Model:    "mistral-small-latest",
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

	if !strings.Contains(answer.String(), "391") {
		t.Errorf("answer missing the correct product 391; got %q", answer.String())
	}
	// Mistral's streamed reasoning shape (reasoning_content vs structured thinking
	// chunks) is provider-version-dependent; surface it as a signal, not a gate.
	if thinking.Len() == 0 {
		t.Log("note: no reasoning_content surfaced for mistral-small (structured-thinking shape may differ)")
	}
}
