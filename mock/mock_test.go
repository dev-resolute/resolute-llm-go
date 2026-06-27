package mock

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/dev-resolute/resolute-llm-go"
)

func TestMockProviderExactMatch(t *testing.T) {
	m := New("mock")
	m.OnPrompt(Exact("hello")).RespondText("world").Add()

	req := llm.LLMRequest{
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "hello"}},
		},
	}

	stream := m.Stream(context.Background(), req)

	var got []llm.LLMEvent
	for ev := range stream.Events {
		got = append(got, ev)
	}
	result := <-stream.Done

	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}

	if len(got) != 2 {
		t.Fatalf("expected 2 events, got %d", len(got))
	}
	if _, ok := got[0].(llm.TextDeltaEvent); !ok {
		t.Fatalf("expected TextDeltaEvent, got %T", got[0])
	}
	if _, ok := got[1].(llm.MessageEndEvent); !ok {
		t.Fatalf("expected MessageEndEvent, got %T", got[1])
	}
}

func TestMockProviderUnmatchedCall(t *testing.T) {
	m := New("mock")
	m.OnPrompt(Exact("hello")).RespondText("world").Add()

	req := llm.LLMRequest{
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "goodbye"}},
		},
	}

	stream := m.Stream(context.Background(), req)
	<-stream.Events
	result := <-stream.Done

	if result.Err == nil {
		t.Fatal("expected error for unmatched call")
	}
	if !errors.Is(result.Err, llm.ErrProviderFatal) {
		t.Fatalf("expected ErrProviderFatal, got %v", result.Err)
	}
}

func TestMockProviderDelay(t *testing.T) {
	m := New("mock")
	m.OnPrompt(Exact("slow")).RespondText("ok").Delay(50 * time.Millisecond).Add()

	req := llm.LLMRequest{
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "slow"}},
		},
	}

	start := time.Now()
	stream := m.Stream(context.Background(), req)
	<-stream.Events
	<-stream.Done
	elapsed := time.Since(start)

	if elapsed < 40*time.Millisecond {
		t.Fatalf("expected delay, got %v", elapsed)
	}
}

func TestMockProviderToolCallResponse(t *testing.T) {
	m := New("mock")
	m.OnPrompt(Exact("call tool")).RespondToolCall("calc", []byte(`{"x":1}`)).Add()

	req := llm.LLMRequest{
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "call tool"}},
		},
	}

	stream := m.Stream(context.Background(), req)
	var got []llm.LLMEvent
	for ev := range stream.Events {
		got = append(got, ev)
	}
	result := <-stream.Done

	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}

	if len(got) != 3 {
		t.Fatalf("expected 3 events, got %d", len(got))
	}
	if _, ok := got[0].(llm.ToolCallStartEvent); !ok {
		t.Fatalf("expected ToolCallStartEvent, got %T", got[0])
	}
	if _, ok := got[1].(llm.ToolCallEndEvent); !ok {
		t.Fatalf("expected ToolCallEndEvent, got %T", got[1])
	}
	if _, ok := got[2].(llm.MessageEndEvent); !ok {
		t.Fatalf("expected MessageEndEvent, got %T", got[2])
	}
}

func TestMockProviderPredicateMatch(t *testing.T) {
	m := New("mock")
	m.OnPrompt(Predicate(func(msgs []llm.Message) bool {
		return len(msgs) > 0
	})).RespondText("yes").Add()

	req := llm.LLMRequest{
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "anything"}},
		},
	}

	stream := m.Stream(context.Background(), req)
	<-stream.Events
	result := <-stream.Done

	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
}

func TestMockProviderStreamError(t *testing.T) {
	m := New("mock")
	m.OnPrompt(Exact("err")).RespondText("partial").StreamError(errors.New("boom")).Add()

	req := llm.LLMRequest{
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "err"}},
		},
	}

	stream := m.Stream(context.Background(), req)
	var got []llm.LLMEvent
	for ev := range stream.Events {
		got = append(got, ev)
	}
	<-stream.Done

	var foundError bool
	for _, ev := range got {
		if ee, ok := ev.(llm.LLMErrorEvent); ok {
			if ee.Error != nil && ee.Error.Error() == "boom" {
				foundError = true
			}
		}
	}
	if !foundError {
		t.Fatal("expected LLMErrorEvent with 'boom'")
	}
}

func TestMockProviderLastUserMatch(t *testing.T) {
	m := New("mock")
	m.OnPrompt(LastUser("second")).RespondText("matched second").Add()

	req := llm.LLMRequest{
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "first"}},
			{Role: "assistant", Content: llm.TextContent{Text: "ok"}},
			{Role: "user", Content: llm.TextContent{Text: "second"}},
		},
	}

	stream := m.Stream(context.Background(), req)
	<-stream.Events
	result := <-stream.Done

	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
}

func TestMockProviderRecordHeaders(t *testing.T) {
	m := New("mock").RecordHeaders()
	m.OnPrompt(Exact("hello")).RespondText("world").Add()

	req := llm.LLMRequest{
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "hello"}},
		},
		Headers: map[string]string{"X-Test": "1"},
	}

	stream := m.Stream(context.Background(), req)
	<-stream.Events
	<-stream.Done

	got := m.RecordedHeaders()
	if got["X-Test"] != "1" {
		t.Fatalf("expected recorded header X-Test=1, got %q", got["X-Test"])
	}
}

func TestMockProviderStatusAndRespHeaders(t *testing.T) {
	var gotStatus int
	var gotHeaders map[string]string
	m := New("mock")
	m.OnPrompt(Exact("hello")).RespondText("world").
		Status(201).
		RespHeaders(map[string]string{"X-Request-ID": "abc"}).
		Add()

	req := llm.LLMRequest{
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "hello"}},
		},
		OnAfterResponse: func(statusCode int, headers map[string]string) {
			gotStatus = statusCode
			gotHeaders = headers
		},
	}

	stream := m.Stream(context.Background(), req)
	<-stream.Events
	<-stream.Done

	if gotStatus != 201 {
		t.Fatalf("expected status 201, got %d", gotStatus)
	}
	if gotHeaders["X-Request-ID"] != "abc" {
		t.Fatalf("expected X-Request-ID=abc, got %q", gotHeaders["X-Request-ID"])
	}
}

func TestMockProviderInvokesOnBeforeRequest(t *testing.T) {
	var called bool
	m := New("mock")
	m.OnPrompt(Exact("hello")).RespondText("world").Add()

	req := llm.LLMRequest{
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "hello"}},
		},
		OnBeforeRequest: func(headers map[string]string) error {
			called = true
			return nil
		},
	}

	stream := m.Stream(context.Background(), req)
	<-stream.Events
	<-stream.Done

	if !called {
		t.Fatal("expected OnBeforeRequest to be called")
	}
}
