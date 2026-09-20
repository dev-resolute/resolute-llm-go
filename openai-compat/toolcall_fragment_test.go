package openaicompat

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

// serveSSE writes the given raw SSE data lines (one per element) plus [DONE].
func serveSSE(t *testing.T, lines ...string) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, l := range lines {
			fmt.Fprintf(w, "data: %s\n\n", l)
			flusher.Flush()
		}
		fmt.Fprintln(w, "data: [DONE]")
	}))
	t.Cleanup(ts.Close)
	return ts
}

// Mistral fragments one logical tool call across chunks whose continuation
// deltas omit the tool-call ID (upstream #8387): the deltas must merge into
// the open call, not split into a second empty-ID call.
func TestFragmentedToolCallContinuationMerges(t *testing.T) {
	ts := serveSSE(t,
		`{"choices":[{"delta":{"tool_calls":[{"id":"call_1","function":{"name":"bash","arguments":"{\"cmd"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"\":\"ls\"}"}}]}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
	)

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	stream := p.Stream(context.Background(), llm.LLMRequest{
		Model:    "mistral-medium-latest",
		Messages: []llm.Message{{Role: "user", Content: llm.TextContent{Text: "hi"}}},
	})
	var ends []llm.ToolCallEndEvent
	for ev := range stream.Events {
		if e, ok := ev.(llm.ToolCallEndEvent); ok {
			ends = append(ends, e)
		}
	}
	result := <-stream.Done
	if result.Err != nil {
		t.Fatalf("stream: %v", result.Err)
	}

	if len(ends) != 1 {
		t.Fatalf("ToolCallEndEvent count = %d, want 1 (fragmented call merged): %+v", len(ends), ends)
	}
	if ends[0].CallID != "call_1" || ends[0].ToolName != "bash" {
		t.Errorf("merged call = %s/%s, want call_1/bash", ends[0].CallID, ends[0].ToolName)
	}
	if got := string(ends[0].Args); got != `{"cmd":"ls"}` {
		t.Errorf("merged args = %q, want %q", got, `{"cmd":"ls"}`)
	}
}

// A chunk carrying only an empty-ID tool-call delta with no call open still
// buffers under the empty ID rather than panicking or dropping (pre-#8387
// fallback path preserved).
func TestEmptyIDToolCallWithoutOpenCallStillBuffers(t *testing.T) {
	ts := serveSSE(t,
		`{"choices":[{"delta":{"tool_calls":[{"function":{"name":"bash","arguments":"{}"}}]}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
	)

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	stream := p.Stream(context.Background(), llm.LLMRequest{Model: "test-model"})
	var ends []llm.ToolCallEndEvent
	for ev := range stream.Events {
		if e, ok := ev.(llm.ToolCallEndEvent); ok {
			ends = append(ends, e)
		}
	}
	if result := <-stream.Done; result.Err != nil {
		t.Fatalf("stream: %v", result.Err)
	}
	if len(ends) != 1 || ends[0].ToolName != "bash" {
		t.Fatalf("ends = %+v, want one bash call", ends)
	}
}
