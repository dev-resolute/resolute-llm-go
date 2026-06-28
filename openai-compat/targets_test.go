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

// captureClassifiedBody runs one stream through a provider wired with the given
// thinking dialect and family classifier (as the built-in constructors do), and
// returns the decoded request body so tests can assert the wire shape per model.
func captureClassifiedBody(t *testing.T, compat Compat, classify func(string) classification, req llm.LLMRequest) map[string]any {
	t.Helper()
	var captured map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &captured)
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	t.Cleanup(ts.Close)

	p, err := newProvider(Config{Name: "target", BaseURL: ts.URL, Compat: compat})
	if err != nil {
		t.Fatalf("newProvider: %v", err)
	}
	p.classify = classify
	stream := p.Stream(context.Background(), req)
	for range stream.Events {
	}
	if res := <-stream.Done; res.Err != nil {
		t.Fatalf("stream: %v", res.Err)
	}
	return captured
}
