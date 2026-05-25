package openaicompat

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/resolute-sh/pi-llm-go"
)

func TestProviderStreamText(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		chunks := []string{
			`{"choices":[{"delta":{"content":"hello"}}]}`,
			`{"choices":[{"delta":{"content":" world"}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := llm.LLMRequest{
		Model: "test-model",
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "hi"}},
		},
	}

	stream := p.Stream(context.Background(), req)
	var got []llm.LLMEvent
	for ev := range stream.Events {
		got = append(got, ev)
	}
	result := <-stream.Done

	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}

	var text string
	for _, ev := range got {
		if td, ok := ev.(llm.TextDeltaEvent); ok {
			text += td.Delta
		}
	}
	if text != "hello world" {
		t.Fatalf("expected 'hello world', got %q", text)
	}
}

func TestProviderStreamToolCall(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		chunks := []string{
			`{"choices":[{"delta":{"tool_calls":[{"id":"call_1","function":{"name":"calc"}}]}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"id":"call_1","function":{"arguments":"{\"x\":1}"}}]}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := llm.LLMRequest{
		Model: "test-model",
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "calc"}},
		},
	}

	stream := p.Stream(context.Background(), req)
	var got []llm.LLMEvent
	for ev := range stream.Events {
		got = append(got, ev)
	}
	result := <-stream.Done

	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}

	var foundStart, foundEnd bool
	for _, ev := range got {
		if _, ok := ev.(llm.ToolCallStartEvent); ok {
			foundStart = true
		}
		if _, ok := ev.(llm.ToolCallEndEvent); ok {
			foundEnd = true
		}
	}
	if !foundStart {
		t.Fatal("expected ToolCallStartEvent")
	}
	if !foundEnd {
		t.Fatal("expected ToolCallEndEvent")
	}
}

func TestProviderHTTPError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer ts.Close()

	p, err := New(Config{BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := llm.LLMRequest{Model: "test-model"}
	stream := p.Stream(context.Background(), req)
	<-stream.Events
	result := <-stream.Done

	if result.Err == nil {
		t.Fatal("expected error for HTTP 401")
	}
	if !strings.Contains(result.Err.Error(), "401") {
		t.Fatalf("expected 401 in error, got %v", result.Err)
	}
}
