package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/resolute-sh/pi-llm-go"
)

func captureReasoningEffort(t *testing.T, thinking llm.ThinkingLevel) (string, bool) {
	t.Helper()
	var body map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	t.Cleanup(ts.Close)

	p, err := New(Config{BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	stream := p.Stream(context.Background(), llm.LLMRequest{Model: "test-model", Thinking: thinking})
	for range stream.Events {
	}
	if result := <-stream.Done; result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}

	effort, ok := body["reasoning_effort"].(string)
	return effort, ok
}

func TestThinkingMinimalMapsToMinimalEffort(t *testing.T) {
	// given a request at the minimal thinking level
	// when the provider builds the wire request
	effort, ok := captureReasoningEffort(t, llm.ThinkingMinimal)

	// then reasoning_effort is "minimal", not "low"
	if !ok {
		t.Fatalf("reasoning_effort absent, want %q", "minimal")
	}
	if effort != "minimal" {
		t.Errorf("reasoning_effort = %q, want %q", effort, "minimal")
	}
}

func TestThinkingBudgetsIgnored(t *testing.T) {
	// given a request that sets ThinkingBudgets (a token-based override)
	var body map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	t.Cleanup(ts.Close)

	p, err := New(Config{BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req := llm.LLMRequest{
		Model:           "o4-mini",
		Thinking:        llm.ThinkingHigh,
		ThinkingBudgets: map[llm.ThinkingLevel]int{llm.ThinkingHigh: 8000},
	}

	// when the provider builds the wire request
	stream := p.Stream(context.Background(), req)
	for range stream.Events {
	}
	if result := <-stream.Done; result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}

	// then reasoning_effort still derives from the level, and no token budget leaks to the wire
	if effort, _ := body["reasoning_effort"].(string); effort != "high" {
		t.Errorf("reasoning_effort = %q, want %q", effort, "high")
	}
	for k := range body {
		if k == "thinking_budget" || k == "thinking_budgets" {
			t.Errorf("token budget %q leaked into the OpenAI-compatible request body", k)
		}
	}
}

func TestThinkingLevelEffortMapping(t *testing.T) {
	tests := []struct {
		name     string
		thinking llm.ThinkingLevel
		want     string
		wantSet  bool
	}{
		{name: "off omits reasoning_effort", thinking: llm.ThinkingOff, wantSet: false},
		{name: "minimal", thinking: llm.ThinkingMinimal, want: "minimal", wantSet: true},
		{name: "low", thinking: llm.ThinkingLow, want: "low", wantSet: true},
		{name: "medium", thinking: llm.ThinkingMedium, want: "medium", wantSet: true},
		{name: "high", thinking: llm.ThinkingHigh, want: "high", wantSet: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			effort, ok := captureReasoningEffort(t, tt.thinking)
			if ok != tt.wantSet {
				t.Fatalf("reasoning_effort present = %v, want %v", ok, tt.wantSet)
			}
			if tt.wantSet && effort != tt.want {
				t.Errorf("reasoning_effort = %q, want %q", effort, tt.want)
			}
		})
	}
}
