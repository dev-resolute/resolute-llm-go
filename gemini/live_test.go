//go:build integration

package gemini

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/resolute-sh/pi-llm-go"
)

func liveProvider(t *testing.T) llm.LLMProvider {
	t.Helper()
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		t.Skip("GEMINI_API_KEY not set")
	}
	p, err := New(Config{APIKey: key})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func streamThinking(t *testing.T, p llm.LLMProvider, model string) (thinking, answer string, err error) {
	t.Helper()
	req := llm.LLMRequest{
		Model:    model,
		Thinking: llm.ThinkingHigh,
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "What is 17 * 23? Reason step by step, then state the final number."}},
		},
	}
	stream := p.Stream(context.Background(), req)
	var think, ans strings.Builder
	for ev := range stream.Events {
		switch e := ev.(type) {
		case llm.ThinkingDeltaEvent:
			think.WriteString(e.Delta)
		case llm.TextDeltaEvent:
			ans.WriteString(e.Delta)
		}
	}
	res := <-stream.Done
	return think.String(), ans.String(), res.Err
}

func TestLiveGeminiThinkingSurfaces_2_5_Flash(t *testing.T) {
	// given a live Gemini 2.5 model with thinking requested
	p := liveProvider(t)

	// when the provider streams a reasoning prompt
	thinking, answer, err := streamThinking(t, p, "gemini-2.5-flash")

	// then thinking surfaces and the full answer streams through correctly
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if thinking == "" {
		t.Error("expected thinking content (IncludeThoughts must surface thoughts), got none")
	}
	if !strings.Contains(answer, "391") {
		t.Errorf("answer missing the correct product 391; got %q", answer)
	}
}

func TestLiveGeminiThinkingSurfaces_3_Flash(t *testing.T) {
	// given a live Gemini 3 model (skips when the key has no access to it)
	p := liveProvider(t)

	// when the provider streams a reasoning prompt
	thinking, answer, err := streamThinking(t, p, "gemini-3.5-flash")
	if err != nil {
		t.Skipf("gemini-3.5-flash not available to this key: %v", err)
	}

	// then thinking surfaces via the thinkingLevel mechanism, not just for 2.5
	if thinking == "" {
		t.Error("expected thinking content for gemini-3 (thinkingLevel mechanism), got none")
	}
	if !strings.Contains(answer, "391") {
		t.Errorf("answer missing the correct product 391; got %q", answer)
	}
}
