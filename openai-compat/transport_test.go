package openaicompat

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

func streamWithTransport(t *testing.T, transport llm.TransportPreference) error {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	t.Cleanup(ts.Close)

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	stream := p.Stream(context.Background(), llm.LLMRequest{Model: "test-model", Transport: transport})
	for range stream.Events {
	}
	return (<-stream.Done).Err
}

func TestWebSocketTransportUnsupported(t *testing.T) {
	// given a request preferring the websocket transport
	// when the SSE-only provider runs it
	err := streamWithTransport(t, llm.TransportWebSocket)

	// then it fails fast with ErrTransportUnsupported
	if !errors.Is(err, llm.ErrTransportUnsupported) {
		t.Errorf("err = %v, want ErrTransportUnsupported", err)
	}
}

func TestDefaultAndSSETransportSucceed(t *testing.T) {
	tests := []struct {
		name      string
		transport llm.TransportPreference
	}{
		{name: "auto (default)", transport: llm.TransportAuto},
		{name: "sse", transport: llm.TransportSSE},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// given a supported transport preference
			// when the provider runs it
			err := streamWithTransport(t, tt.transport)

			// then the call succeeds over HTTP/SSE
			if err != nil {
				t.Errorf("err = %v, want nil for %s", err, tt.transport)
			}
		})
	}
}
