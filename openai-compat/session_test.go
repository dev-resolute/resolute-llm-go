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

type capturedRequest struct {
	headers http.Header
	body    map[string]any
}

func captureRequest(t *testing.T, req llm.LLMRequest) capturedRequest {
	t.Helper()
	var cap capturedRequest
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.headers = r.Header.Clone()
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &cap.body)
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	t.Cleanup(ts.Close)

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	req.Model = "test-model"
	stream := p.Stream(context.Background(), req)
	for range stream.Events {
	}
	if result := <-stream.Done; result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
	return cap
}

func TestSessionIDSetsAffinityHeadersAndCacheKey(t *testing.T) {
	// given a request carrying a session id
	cap := captureRequest(t, llm.LLMRequest{SessionID: "sess-abc"})

	// then the affinity headers and prompt_cache_key are populated from it
	for _, h := range []string{"Session_id", "X-Client-Request-Id", "X-Session-Affinity"} {
		if got := cap.headers.Get(h); got != "sess-abc" {
			t.Errorf("header %s = %q, want %q", h, got, "sess-abc")
		}
	}
	if got, _ := cap.body["prompt_cache_key"].(string); got != "sess-abc" {
		t.Errorf("prompt_cache_key = %q, want %q", got, "sess-abc")
	}
}

func TestPromptCacheKeyClampedToMaxLength(t *testing.T) {
	// given a session id longer than the OpenAI prompt-cache-key limit (64)
	long := ""
	for i := 0; i < 100; i++ {
		long += "a"
	}
	cap := captureRequest(t, llm.LLMRequest{SessionID: long})

	// then prompt_cache_key is truncated to 64 chars
	got, _ := cap.body["prompt_cache_key"].(string)
	if len(got) != 64 {
		t.Errorf("prompt_cache_key length = %d, want 64", len(got))
	}
	// but the affinity headers keep the full value (upstream clamps only the body key)
	if h := cap.headers.Get("X-Session-Affinity"); h != long {
		t.Errorf("x-session-affinity = %q (len %d), want full session id (len %d)", h, len(h), len(long))
	}
}

func TestEmptySessionIDOmitsAffinity(t *testing.T) {
	// given a request with no session id
	cap := captureRequest(t, llm.LLMRequest{})

	// then no affinity headers and no prompt_cache_key are sent
	for _, h := range []string{"Session_id", "X-Client-Request-Id", "X-Session-Affinity"} {
		if got := cap.headers.Get(h); got != "" {
			t.Errorf("header %s = %q, want empty", h, got)
		}
	}
	if _, present := cap.body["prompt_cache_key"]; present {
		t.Errorf("prompt_cache_key present, want absent")
	}
}
