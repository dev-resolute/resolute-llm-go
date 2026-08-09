package openaicompat

import (
	"context"
	"encoding/json"
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

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
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
	var msgEnd llm.MessageEndEvent
	for _, ev := range got {
		if td, ok := ev.(llm.TextDeltaEvent); ok {
			text += td.Delta
		}
		if e, ok := ev.(llm.MessageEndEvent); ok {
			msgEnd = e
		}
	}
	if text != "hello world" {
		t.Fatalf("expected 'hello world', got %q", text)
	}
	if msgEnd.StopReason != llm.StopReasonStop {
		t.Errorf("StopReason = %q, want stop", msgEnd.StopReason)
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

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
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

	var end llm.ToolCallEndEvent
	var msgEnd llm.MessageEndEvent
	for _, ev := range got {
		switch e := ev.(type) {
		case llm.ToolCallEndEvent:
			end = e
		case llm.MessageEndEvent:
			msgEnd = e
		}
	}
	if end.ToolName != "calc" || string(end.Args) != `{"x":1}` {
		t.Errorf("ToolCallEndEvent = %+v, want finalized name calc and args {\"x\":1}", end)
	}
	if msgEnd.StopReason != llm.StopReasonToolUse {
		t.Errorf("StopReason = %q, want toolUse", msgEnd.StopReason)
	}
}

func TestProviderStreamMultipleToolCallsPreserveOrder(t *testing.T) {
	// Regression test: readSSE used to flush buffered tool calls by ranging
	// over a map, which randomizes emission order for multi-call turns.
	// Upstream preserves content order (calls appear in the order the model
	// emitted them), and downstream (agent-core) now derives execution order
	// from ToolCallEndEvent, so this must be deterministic.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		chunks := []string{
			`{"choices":[{"delta":{"tool_calls":[{"id":"call_1","function":{"name":"a"}}]}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"id":"call_2","function":{"name":"b"}}]}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"id":"call_3","function":{"name":"c"}}]}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"id":"call_1","function":{"arguments":"{\"x\":1}"}}]}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"id":"call_2","function":{"arguments":"{\"y\":2}"}}]}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"id":"call_3","function":{"arguments":"{\"z\":3}"}}]}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
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

	wantOrder := []struct {
		callID   string
		toolName string
		args     string
	}{
		{"call_1", "a", `{"x":1}`},
		{"call_2", "b", `{"y":2}`},
		{"call_3", "c", `{"z":3}`},
	}

	var ends []llm.ToolCallEndEvent
	for _, ev := range got {
		if e, ok := ev.(llm.ToolCallEndEvent); ok {
			ends = append(ends, e)
		}
	}
	if len(ends) != len(wantOrder) {
		t.Fatalf("expected %d ToolCallEndEvents, got %d: %+v", len(wantOrder), len(ends), ends)
	}
	for i, want := range wantOrder {
		if ends[i].CallID != want.callID || ends[i].ToolName != want.toolName || string(ends[i].Args) != want.args {
			t.Errorf("ToolCallEndEvent[%d] = %+v, want CallID=%s ToolName=%s Args=%s", i, ends[i], want.callID, want.toolName, want.args)
		}
	}

	var toolCallMsgs []llm.ToolCallContent
	for _, msg := range result.Messages {
		if tc, ok := msg.Content.(llm.ToolCallContent); ok {
			toolCallMsgs = append(toolCallMsgs, tc)
		}
	}
	if len(toolCallMsgs) != len(wantOrder) {
		t.Fatalf("expected %d tool-call messages, got %d: %+v", len(wantOrder), len(toolCallMsgs), toolCallMsgs)
	}
	for i, want := range wantOrder {
		if toolCallMsgs[i].CallID != want.callID || toolCallMsgs[i].ToolName != want.toolName || string(toolCallMsgs[i].Args) != want.args {
			t.Errorf("Messages tool-call[%d] = %+v, want CallID=%s ToolName=%s Args=%s", i, toolCallMsgs[i], want.callID, want.toolName, want.args)
		}
	}
}

func TestProviderStreamToolCallTruncatedByLength(t *testing.T) {
	// finish_reason "length" mid tool call: the buffered call must still be
	// delivered (ToolCallEndEvent with whatever args arrived) and the message
	// end must report StopReasonLength so the agent layer can refuse to run it.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		chunks := []string{
			`{"choices":[{"delta":{"tool_calls":[{"id":"call_1","function":{"name":"calc"}}]}}]}`,
			`{"choices":[{"delta":{"tool_calls":[{"id":"call_1","function":{"arguments":"{\"x\":"}}]}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"length"}]}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
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

	var ends []llm.ToolCallEndEvent
	var msgEnd llm.MessageEndEvent
	for _, ev := range got {
		switch e := ev.(type) {
		case llm.ToolCallEndEvent:
			ends = append(ends, e)
		case llm.MessageEndEvent:
			msgEnd = e
		}
	}
	if len(ends) != 1 {
		t.Fatalf("expected exactly one ToolCallEndEvent, got %d: %+v", len(ends), ends)
	}
	want := llm.ToolCallEndEvent{CallID: "call_1", ToolName: "calc", Args: json.RawMessage(`{"x":`)}
	if ends[0].CallID != want.CallID || ends[0].ToolName != want.ToolName || string(ends[0].Args) != string(want.Args) {
		t.Errorf("ToolCallEndEvent = %+v, want %+v", ends[0], want)
	}
	if msgEnd.StopReason != llm.StopReasonLength {
		t.Errorf("StopReason = %q, want length", msgEnd.StopReason)
	}
}

func TestProviderHTTPError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer ts.Close()

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
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
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
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
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{Name: "openai-compat",
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
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{Name: "openai-compat",
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
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{Name: "openai-compat",
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
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{Name: "openai-compat",
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
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	p, err := New(Config{Name: "openai-compat",
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
	p, err := New(Config{Name: "openai-compat",
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

	p, err := New(Config{Name: "openai-compat",
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
