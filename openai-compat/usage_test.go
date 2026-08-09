package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

func TestMapUsageChunk(t *testing.T) {
	for _, tt := range []struct {
		name string
		u    *usageChunk
		want llm.UsageEvent
	}{
		{"plain", &usageChunk{PromptTokens: 100, CompletionTokens: 42}, llm.UsageEvent{InputTokens: 100, OutputTokens: 42}},
		{"cached subtracted", &usageChunk{
			PromptTokens: 100, CompletionTokens: 42,
			PromptTokensDetails: &usagePromptDetails{CachedTokens: 30},
		}, llm.UsageEvent{InputTokens: 70, OutputTokens: 42}},
		{"cache write subtracted", &usageChunk{
			PromptTokens: 100, CompletionTokens: 42,
			PromptTokensDetails: &usagePromptDetails{CachedTokens: 10, CacheWriteTokens: 20},
		}, llm.UsageEvent{InputTokens: 70, OutputTokens: 42}},
		{"legacy hit field", &usageChunk{PromptTokens: 100, CompletionTokens: 42, PromptCacheHitTokens: 25},
			llm.UsageEvent{InputTokens: 75, OutputTokens: 42}},
		{"floor at zero", &usageChunk{
			PromptTokens: 10, CompletionTokens: 42,
			PromptTokensDetails: &usagePromptDetails{CachedTokens: 50},
		}, llm.UsageEvent{InputTokens: 0, OutputTokens: 42}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := mapUsageChunk(tt.u); got != tt.want {
				t.Errorf("mapUsageChunk(%+v) = %+v, want %+v", tt.u, got, tt.want)
			}
		})
	}
}

// The provider asks for streamed usage by default (upstream
// stream_options.include_usage) and parses it whenever it arrives.
func TestProviderRequestsUsageInStreamOptions(t *testing.T) {
	var body map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	t.Cleanup(ts.Close)

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	stream := p.Stream(context.Background(), llm.LLMRequest{
		Model:    "test-model",
		Messages: []llm.Message{{Role: "user", Content: llm.TextContent{Text: "hi"}}},
	})
	for range stream.Events {
	}
	if res := <-stream.Done; res.Err != nil {
		t.Fatalf("stream: %v", res.Err)
	}

	opts, ok := body["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf("stream_options missing from request body: %v", body)
	}
	if opts["include_usage"] != true {
		t.Errorf("stream_options.include_usage = %v, want true", opts["include_usage"])
	}
}

func TestProviderOmitsStreamOptionsWhenUsageUnsupported(t *testing.T) {
	var body map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	t.Cleanup(ts.Close)

	p, err := New(Config{
		Name:    "openai-compat",
		BaseURL: ts.URL,
		Compat:  Compat{SupportsUsageInStreaming: boolPtr(false)},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	stream := p.Stream(context.Background(), llm.LLMRequest{
		Model:    "test-model",
		Messages: []llm.Message{{Role: "user", Content: llm.TextContent{Text: "hi"}}},
	})
	for range stream.Events {
	}
	if res := <-stream.Done; res.Err != nil {
		t.Fatalf("stream: %v", res.Err)
	}

	if _, present := body["stream_options"]; present {
		t.Errorf("stream_options present despite SupportsUsageInStreaming=false: %v", body["stream_options"])
	}
}

// At most one UsageEvent per stream, carrying the request's final totals,
// emitted before MessageEndEvent.
func TestProviderUsageEventFromFinalChunk(t *testing.T) {
	ts := sseServer(t, []string{
		`{"choices":[{"delta":{"content":"hello"}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		`{"choices":[],"usage":{"prompt_tokens":110,"completion_tokens":42,"prompt_tokens_details":{"cached_tokens":10}}}`,
	})

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, result := streamAndCollect(t, p)
	if result.Err != nil {
		t.Fatalf("result.Err = %v, want nil", result.Err)
	}

	var usageEvents []llm.UsageEvent
	var usageIdx, msgEndIdx = -1, -1
	for i, ev := range events {
		if u, ok := ev.(llm.UsageEvent); ok {
			usageEvents = append(usageEvents, u)
			usageIdx = i
		}
		if _, ok := ev.(llm.MessageEndEvent); ok {
			msgEndIdx = i
		}
	}
	if len(usageEvents) != 1 {
		t.Fatalf("UsageEvents = %d, want exactly 1", len(usageEvents))
	}
	want := llm.UsageEvent{InputTokens: 100, OutputTokens: 42}
	if usageEvents[0] != want {
		t.Errorf("UsageEvent = %+v, want %+v", usageEvents[0], want)
	}
	if msgEndIdx == -1 || usageIdx > msgEndIdx {
		t.Errorf("UsageEvent index %d, MessageEndEvent index %d: usage must precede message end", usageIdx, msgEndIdx)
	}
}

// Moonshot-style servers report usage on the choice instead of the chunk
// (upstream fallback).
func TestProviderUsageEventChoiceFallback(t *testing.T) {
	ts := sseServer(t, []string{
		`{"choices":[{"delta":{"content":"hi"},"usage":{"prompt_tokens":50,"completion_tokens":7}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
	})

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, result := streamAndCollect(t, p)
	if result.Err != nil {
		t.Fatalf("result.Err = %v, want nil", result.Err)
	}

	var usageEvents []llm.UsageEvent
	for _, ev := range events {
		if u, ok := ev.(llm.UsageEvent); ok {
			usageEvents = append(usageEvents, u)
		}
	}
	if len(usageEvents) != 1 {
		t.Fatalf("UsageEvents = %d, want exactly 1", len(usageEvents))
	}
	want := llm.UsageEvent{InputTokens: 50, OutputTokens: 7}
	if usageEvents[0] != want {
		t.Errorf("UsageEvent = %+v, want %+v", usageEvents[0], want)
	}
}

// Last report wins when several chunks carry usage.
func TestProviderUsageEventLastReportWins(t *testing.T) {
	ts := sseServer(t, []string{
		`{"choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":50,"completion_tokens":3}}`,
		`{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":60,"completion_tokens":4}}`,
	})

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, result := streamAndCollect(t, p)
	if result.Err != nil {
		t.Fatalf("result.Err = %v, want nil", result.Err)
	}

	var usageEvents []llm.UsageEvent
	for _, ev := range events {
		if u, ok := ev.(llm.UsageEvent); ok {
			usageEvents = append(usageEvents, u)
		}
	}
	if len(usageEvents) != 1 {
		t.Fatalf("UsageEvents = %d, want exactly 1", len(usageEvents))
	}
	want := llm.UsageEvent{InputTokens: 60, OutputTokens: 4}
	if usageEvents[0] != want {
		t.Errorf("UsageEvent = %+v, want %+v (last report)", usageEvents[0], want)
	}
}

func TestProviderNoUsageEventWhenAbsent(t *testing.T) {
	ts := sseServer(t, []string{
		`{"choices":[{"delta":{"content":"hello"}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
	})

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, result := streamAndCollect(t, p)
	if result.Err != nil {
		t.Fatalf("result.Err = %v, want nil", result.Err)
	}

	for _, ev := range events {
		if _, ok := ev.(llm.UsageEvent); ok {
			t.Error("UsageEvent emitted for a stream with no usage report")
		}
	}
}
