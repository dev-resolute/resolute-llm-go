package openaicompat

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
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

func TestProviderPerCallHeaders(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Test") != "from-request" {
			t.Errorf("expected X-Test=from-request, got %q", r.Header.Get("X-Test"))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := llm.LLMRequest{
		Model:   "test-model",
		Headers: map[string]string{"X-Test": "from-request"},
	}
	stream := p.Stream(context.Background(), req)
	<-stream.Events
	result := <-stream.Done
	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
}

func TestProviderPerCallHeaderWinsOnConflict(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Test") != "from-request" {
			t.Errorf("expected X-Test=from-request, got %q", r.Header.Get("X-Test"))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{
		BaseURL: ts.URL,
		Headers: map[string]string{"X-Test": "from-config"},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := llm.LLMRequest{
		Model:   "test-model",
		Headers: map[string]string{"X-Test": "from-request"},
	}
	stream := p.Stream(context.Background(), req)
	<-stream.Events
	result := <-stream.Done
	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
}

func TestProviderGetAPIKeyCalledOnce(t *testing.T) {
	var count int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{
		BaseURL: ts.URL,
		GetAPIKey: func(ctx context.Context) (string, error) {
			count++
			return "refresh-token", nil
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := llm.LLMRequest{Model: "test-model"}
	stream := p.Stream(context.Background(), req)
	<-stream.Events
	result := <-stream.Done
	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
	if count != 1 {
		t.Fatalf("expected GetAPIKey called once, got %d", count)
	}
}

func TestProviderGetAPIKeyUsedAsAuth(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth != "Bearer refresh-token-abc" {
			t.Errorf("expected Authorization=Bearer refresh-token-abc, got %q", auth)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{
		BaseURL: ts.URL,
		GetAPIKey: func(ctx context.Context) (string, error) {
			return "refresh-token-abc", nil
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := llm.LLMRequest{Model: "test-model"}
	stream := p.Stream(context.Background(), req)
	<-stream.Events
	result := <-stream.Done
	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
}

func TestProviderGetAPIKeyNilFallsBackToStatic(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth != "Bearer static" {
			t.Errorf("expected Authorization=Bearer static, got %q", auth)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{
		BaseURL: ts.URL,
		APIKey:  "static",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := llm.LLMRequest{Model: "test-model"}
	stream := p.Stream(context.Background(), req)
	<-stream.Events
	result := <-stream.Done
	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
}

func TestProviderGetAPIKeyEmptyStringFallsBack(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth != "Bearer static" {
			t.Errorf("expected Authorization=Bearer static, got %q", auth)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{
		BaseURL: ts.URL,
		APIKey:  "static",
		GetAPIKey: func(ctx context.Context) (string, error) {
			return "", nil
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := llm.LLMRequest{Model: "test-model"}
	stream := p.Stream(context.Background(), req)
	<-stream.Events
	result := <-stream.Done
	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
}

func TestProviderGetAPIKeyErrorAborts(t *testing.T) {
	var hits int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
	}))
	defer ts.Close()

	wantErr := errors.New("expired refresh token")
	p, err := New(Config{
		BaseURL: ts.URL,
		GetAPIKey: func(ctx context.Context) (string, error) {
			return "", wantErr
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := llm.LLMRequest{Model: "test-model"}
	stream := p.Stream(context.Background(), req)
	<-stream.Events
	result := <-stream.Done

	if result.Err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(result.Err, wantErr) {
		t.Fatalf("expected %v, got %v", wantErr, result.Err)
	}
	if hits > 0 {
		t.Fatal("expected no HTTP request when GetAPIKey errors")
	}
}

func TestProviderGetAPIKeyHonorsContext(t *testing.T) {
	var hits int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
	}))
	defer ts.Close()

	ctx, cancel := context.WithCancelCause(context.Background())
	cancel(errors.New("ctx cancelled"))

	p, err := New(Config{
		BaseURL: ts.URL,
		GetAPIKey: func(ctx context.Context) (string, error) {
			if ctx.Err() != nil {
				return "", ctx.Err()
			}
			return "key", nil
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := llm.LLMRequest{Model: "test-model"}
	stream := p.Stream(ctx, req)
	<-stream.Events
	result := <-stream.Done

	if result.Err == nil {
		t.Fatal("expected error")
	}
	if hits > 0 {
		t.Fatal("expected no HTTP request when ctx is cancelled")
	}
}
