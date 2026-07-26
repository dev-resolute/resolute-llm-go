//go:build integration

package gemini

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
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

func TestLiveGemini3ToolCallThoughtSignatureRoundTrip(t *testing.T) {
	// given a live Gemini 3 model and a weather tool
	p := liveProvider(t)
	const model = "gemini-3.1-pro-preview"
	weather := llm.ToolDef{
		Name:        "get_weather",
		Description: "Get the current weather for a city.",
		Schema:      []byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}
	ask := llm.Message{Role: "user", Content: llm.TextContent{Text: "What is the weather in Paris right now? Use the tool."}}

	// when the model calls the tool
	turn1 := p.Stream(context.Background(), llm.LLMRequest{
		Model:    model,
		Thinking: llm.ThinkingLow,
		Tools:    []llm.ToolDef{weather},
		Messages: []llm.Message{ask},
	})
	var startSig []byte
	for ev := range turn1.Events {
		if e, ok := ev.(llm.ToolCallStartEvent); ok {
			startSig = e.ThoughtSignature
		}
	}
	res := <-turn1.Done
	if res.Err != nil {
		t.Skipf("%s not available to this key: %v", model, res.Err)
	}

	// then the tool call surfaces in the result messages with its thought signature
	var call *llm.ToolCallContent
	for _, m := range res.Messages {
		if tc, ok := m.Content.(llm.ToolCallContent); ok {
			call = &tc
		}
	}
	if call == nil {
		t.Fatal("no ToolCallContent in StreamResult.Messages (tool call lost across chunks)")
	}
	if len(call.ThoughtSignature) == 0 {
		t.Error("ToolCallContent.ThoughtSignature empty; Gemini 3 requires it for replay")
	}
	if len(startSig) == 0 {
		t.Error("ToolCallStartEvent.ThoughtSignature empty; event consumers cannot persist it")
	}

	// and replaying the tool call with its signature completes the turn
	turn2 := p.Stream(context.Background(), llm.LLMRequest{
		Model:    model,
		Thinking: llm.ThinkingLow,
		Tools:    []llm.ToolDef{weather},
		Messages: []llm.Message{
			ask,
			{Role: "assistant", Content: *call},
			{Role: "tool", Content: llm.ToolResultContent{
				CallID:   call.CallID,
				ToolName: call.ToolName,
				Content:  `{"temperature_c": 22, "condition": "sunny"}`,
			}},
		},
	})
	var answer strings.Builder
	for ev := range turn2.Events {
		if e, ok := ev.(llm.TextDeltaEvent); ok {
			answer.WriteString(e.Delta)
		}
	}
	res2 := <-turn2.Done
	if res2.Err != nil {
		t.Fatalf("turn 2 rejected (thought signature not accepted): %v", res2.Err)
	}
	if !strings.Contains(answer.String(), "22") {
		t.Errorf("answer does not use the tool result (want mention of 22); got %q", answer.String())
	}
}

func TestLiveGeminiStrictPreferToolRoundTrip(t *testing.T) {
	// given a live Gemini 3 model and a "prefer" strict tool
	p := liveProvider(t)
	const model = "gemini-3.1-pro-preview"
	weather := llm.ToolDef{
		Name:                "get_weather",
		Description:         "Get the current weather for a city.",
		Schema:              []byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
		ConstrainedSampling: &llm.ConstrainedSampling{Strict: llm.StrictPrefer},
	}
	ask := llm.Message{Role: "user", Content: llm.TextContent{Text: "What is the weather in Paris right now? Use the tool."}}

	// when the model streams against a "prefer" strict tool (VALIDATED
	// function-calling mode, per LLM-12)
	stream := p.Stream(context.Background(), llm.LLMRequest{
		Model:    model,
		Thinking: llm.ThinkingLow,
		Tools:    []llm.ToolDef{weather},
		Messages: []llm.Message{ask},
	})
	for range stream.Events {
	}
	res := <-stream.Done
	if res.Err != nil {
		t.Skipf("%s not available to this key: %v", model, res.Err)
	}

	// then the tool call arrives in the result messages
	var call *llm.ToolCallContent
	for _, m := range res.Messages {
		if tc, ok := m.Content.(llm.ToolCallContent); ok {
			call = &tc
		}
	}
	if call == nil {
		t.Fatal("no ToolCallContent in StreamResult.Messages (strict prefer tool call did not arrive)")
	}
	if call.ToolName != weather.Name {
		t.Errorf("ToolCallContent.ToolName = %q, want %q", call.ToolName, weather.Name)
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
