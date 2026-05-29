package gemini

import (
	"context"
	"errors"
	"testing"

	"github.com/resolute-sh/pi-llm-go"
)

func TestWebSocketTransportUnsupported(t *testing.T) {
	// given a Gemini provider and a websocket transport preference
	p, err := New(Config{APIKey: "test-key"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// when a request prefers websocket
	stream := p.Stream(context.Background(), llm.LLMRequest{
		Model:     "gemini-2.5-flash",
		Transport: llm.TransportWebSocket,
	})
	for range stream.Events {
	}
	result := <-stream.Done

	// then it fails fast with ErrTransportUnsupported, before any API call
	if !errors.Is(result.Err, llm.ErrTransportUnsupported) {
		t.Errorf("err = %v, want ErrTransportUnsupported", result.Err)
	}
}
