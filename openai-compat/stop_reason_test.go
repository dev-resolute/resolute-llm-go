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

func TestCompatStopReason(t *testing.T) {
	for _, tt := range []struct {
		name                 string
		finishReason         string
		sawToolCalls         bool
		supportsFinishReason bool
		want                 llm.StopReason
		wantErr              error
	}{
		{"stop", "stop", false, true, llm.StopReasonStop, nil},
		{"end alias", "end", false, true, llm.StopReasonStop, nil},
		// Providers that report "stop" despite emitting tool calls (broken
		// servers) still map to toolUse.
		{"stop with tool calls", "stop", true, true, llm.StopReasonToolUse, nil},
		{"tool_calls", "tool_calls", true, true, llm.StopReasonToolUse, nil},
		{"function_call", "function_call", true, true, llm.StopReasonToolUse, nil},
		{"length", "length", false, true, llm.StopReasonLength, nil},
		// Length wins over tool calls: truncated arguments may be incomplete
		// (upstream #6285).
		{"length with tool calls", "length", true, true, llm.StopReasonLength, nil},
		{"content_filter", "content_filter", false, true, "", llm.ErrProviderStop},
		{"network_error", "network_error", false, true, "", llm.ErrProviderStop},
		{"unmapped reason", "alien_reason", false, true, "", llm.ErrProviderStop},
		// content_filter with tool calls in flight is still an error: the
		// calls may be cut off mid-arguments, so a toolUse mapping would
		// execute potentially borked calls.
		{"content_filter with tool calls", "content_filter", true, true, "", llm.ErrProviderStop},
		// Missing finish_reason: hard error for providers expected to send
		// one; inferred for providers known to omit it.
		{"missing, supported", "", false, true, "", llm.ErrMalformedResponse},
		{"missing, omitted, text only", "", false, false, llm.StopReasonStop, nil},
		{"missing, omitted, tool calls", "", true, false, llm.StopReasonToolUse, nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := compatStopReason(tt.finishReason, tt.sawToolCalls, tt.supportsFinishReason)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Errorf("compatStopReason(%q, %v, %v) err = %v, want errors.Is %v",
						tt.finishReason, tt.sawToolCalls, tt.supportsFinishReason, err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("compatStopReason(%q, %v, %v) unexpected err = %v",
					tt.finishReason, tt.sawToolCalls, tt.supportsFinishReason, err)
			}
			if got != tt.want {
				t.Errorf("compatStopReason(%q, %v, %v) = %q, want %q",
					tt.finishReason, tt.sawToolCalls, tt.supportsFinishReason, got, tt.want)
			}
		})
	}
}

// sseServer streams the given chunks as SSE data lines followed by [DONE].
func sseServer(t *testing.T, chunks []string) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
		fmt.Fprintln(w, "data: [DONE]")
	}))
	t.Cleanup(ts.Close)
	return ts
}

func streamAndCollect(t *testing.T, p llm.LLMProvider) ([]llm.LLMEvent, llm.StreamResult) {
	t.Helper()
	stream := p.Stream(context.Background(), llm.LLMRequest{
		Model:    "test-model",
		Messages: []llm.Message{{Role: "user", Content: llm.TextContent{Text: "hi"}}},
	})
	var events []llm.LLMEvent
	for ev := range stream.Events {
		events = append(events, ev)
	}
	return events, <-stream.Done
}

// #7272: a terminal finish_reason without a portable mapping (content_filter
// and friends) surfaces as a fatal provider error, not a successful stop.
func TestProviderStreamContentFilterErrors(t *testing.T) {
	ts := sseServer(t, []string{
		`{"choices":[{"delta":{"content":"partial"}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"content_filter"}]}`,
	})

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, result := streamAndCollect(t, p)

	if !errors.Is(result.Err, llm.ErrProviderStop) {
		t.Errorf("result.Err = %v, want errors.Is ErrProviderStop", result.Err)
	}
	var llmErr *llm.LLMErrorEvent
	for _, ev := range events {
		if e, ok := ev.(llm.LLMErrorEvent); ok {
			llmErr = &e
		}
		if _, ok := ev.(llm.MessageEndEvent); ok {
			t.Error("MessageEndEvent emitted for an error-terminated message")
		}
	}
	if llmErr == nil {
		t.Fatal("no LLMErrorEvent emitted")
	}
	if llmErr.Transient {
		t.Error("LLMErrorEvent.Transient = true, want false (fatal)")
	}
	if !errors.Is(llmErr.Error, llm.ErrProviderStop) {
		t.Errorf("LLMErrorEvent.Error = %v, want errors.Is ErrProviderStop", llmErr.Error)
	}
}

func TestProviderStreamUnknownFinishReasonErrors(t *testing.T) {
	ts := sseServer(t, []string{
		`{"choices":[{"delta":{"content":"partial"}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"alien_reason"}]}`,
	})

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, result := streamAndCollect(t, p)

	if !errors.Is(result.Err, llm.ErrProviderStop) {
		t.Errorf("result.Err = %v, want errors.Is ErrProviderStop", result.Err)
	}
}

// supportsFinishReason (upstream 0.84.0): a stream from a provider expected to
// send finish_reason that ends without one is a protocol error, not a silent
// unknown stop.
func TestProviderStreamMissingFinishReasonErrors(t *testing.T) {
	ts := sseServer(t, []string{
		`{"choices":[{"delta":{"content":"hello"}}]}`,
	})

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, result := streamAndCollect(t, p)

	if !errors.Is(result.Err, llm.ErrMalformedResponse) {
		t.Errorf("result.Err = %v, want errors.Is ErrMalformedResponse", result.Err)
	}
	for _, ev := range events {
		if _, ok := ev.(llm.MessageEndEvent); ok {
			t.Error("MessageEndEvent emitted for a stream missing finish_reason")
		}
	}
}

// Providers known to omit finish_reason (Compat.SupportsFinishReason = false)
// get upstream's inference: stop when no tool calls streamed.
func TestProviderStreamMissingFinishReasonInferredStop(t *testing.T) {
	ts := sseServer(t, []string{
		`{"choices":[{"delta":{"content":"hello"}}]}`,
	})

	p, err := New(Config{
		Name:    "openai-compat",
		BaseURL: ts.URL,
		Compat:  Compat{SupportsFinishReason: boolPtr(false)},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, result := streamAndCollect(t, p)

	if result.Err != nil {
		t.Fatalf("result.Err = %v, want nil (inference is not an error)", result.Err)
	}
	var msgEnd *llm.MessageEndEvent
	for _, ev := range events {
		if e, ok := ev.(llm.MessageEndEvent); ok {
			msgEnd = &e
		}
	}
	if msgEnd == nil {
		t.Fatal("no MessageEndEvent emitted")
	}
	if msgEnd.StopReason != llm.StopReasonStop {
		t.Errorf("StopReason = %q, want stop (inferred)", msgEnd.StopReason)
	}
}

// The inference path must also flush tool calls still buffered when the stream
// ends — otherwise providers without finish_reason lose their tool calls.
func TestProviderStreamMissingFinishReasonInfersToolUseAndFlushes(t *testing.T) {
	ts := sseServer(t, []string{
		`{"choices":[{"delta":{"tool_calls":[{"id":"call_1","function":{"name":"calc"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"id":"call_1","function":{"arguments":"{\"x\":1}"}}]}}]}`,
	})

	p, err := New(Config{
		Name:    "openai-compat",
		BaseURL: ts.URL,
		Compat:  Compat{SupportsFinishReason: boolPtr(false)},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, result := streamAndCollect(t, p)

	if result.Err != nil {
		t.Fatalf("result.Err = %v, want nil", result.Err)
	}

	// then the buffered call flushed with its finalized arguments
	var end *llm.ToolCallEndEvent
	for _, ev := range events {
		if e, ok := ev.(llm.ToolCallEndEvent); ok {
			end = &e
		}
	}
	if end == nil {
		t.Fatal("no ToolCallEndEvent emitted (buffered call dropped)")
	}
	if end.CallID != "call_1" || string(end.Args) != `{"x":1}` {
		t.Errorf("ToolCallEndEvent = {CallID: %q, Args: %s}, want {call_1, {\"x\":1}}", end.CallID, end.Args)
	}

	var msgEnd *llm.MessageEndEvent
	for _, ev := range events {
		if e, ok := ev.(llm.MessageEndEvent); ok {
			msgEnd = &e
		}
	}
	if msgEnd == nil || msgEnd.StopReason != llm.StopReasonToolUse {
		t.Errorf("MessageEndEvent = %+v, want StopReason toolUse (inferred)", msgEnd)
	}

	var sawCall bool
	for _, m := range result.Messages {
		if tc, ok := m.Content.(llm.ToolCallContent); ok && tc.CallID == "call_1" {
			sawCall = true
		}
	}
	if !sawCall {
		t.Error("result messages missing the flushed ToolCallContent")
	}
}

func boolPtr(b bool) *bool { return &b }
