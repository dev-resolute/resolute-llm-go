package llm

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestRunEmitsEvents(t *testing.T) {
	req := LLMRequest{
		Model: "test",
		Messages: []Message{
			{Role: "user", Content: TextContent{Text: "hello"}},
		},
	}

	produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
		for i := 0; i < 5; i++ {
			if err := emit(TextDeltaEvent{Delta: fmt.Sprintf("%d", i)}); err != nil {
				return nil, err
			}
		}
		return []Message{{Role: "assistant", Content: TextContent{Text: "01234"}}}, nil
	}

	stream := Run(context.Background(), req, produce)

	var got []LLMEvent
	for ev := range stream.Events {
		got = append(got, ev)
	}
	result := <-stream.Done

	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
	if len(got) != 5 {
		t.Fatalf("expected 5 events, got %d", len(got))
	}
	for i, ev := range got {
		td, ok := ev.(TextDeltaEvent)
		if !ok {
			t.Fatalf("expected TextDeltaEvent, got %T", ev)
		}
		if td.Delta != fmt.Sprintf("%d", i) {
			t.Fatalf("expected delta %d, got %s", i, td.Delta)
		}
	}
	expectedMsgs := append([]Message(nil), req.Messages...)
	expectedMsgs = append(expectedMsgs, Message{Role: "assistant", Content: TextContent{Text: "01234"}})
	if len(result.Messages) != len(expectedMsgs) {
		t.Fatalf("expected %d messages, got %d", len(expectedMsgs), len(result.Messages))
	}
}

func TestRunProducerReturnsError(t *testing.T) {
	req := LLMRequest{Model: "test"}
	wantErr := errors.New("boom")

	produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
		if err := emit(TextDeltaEvent{Delta: "partial"}); err != nil {
			return nil, err
		}
		return nil, wantErr
	}

	stream := Run(context.Background(), req, produce)

	var got []LLMEvent
	for ev := range stream.Events {
		got = append(got, ev)
	}
	result := <-stream.Done

	if result.Err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(result.Err, wantErr) {
		t.Fatalf("expected %v, got %v", wantErr, result.Err)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 event before error, got %d", len(got))
	}
}

func TestRunNilMessages(t *testing.T) {
	req := LLMRequest{
		Model: "test",
		Messages: []Message{
			{Role: "user", Content: TextContent{Text: "hi"}},
		},
	}

	produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
		return nil, nil
	}

	stream := Run(context.Background(), req, produce)

	var got []LLMEvent
	for ev := range stream.Events {
		got = append(got, ev)
	}
	result := <-stream.Done

	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
	if len(got) != 0 {
		t.Fatalf("expected 0 events, got %d", len(got))
	}
	if len(result.Messages) != len(req.Messages) {
		t.Fatalf("expected %d messages, got %d", len(req.Messages), len(result.Messages))
	}
}

func TestRunCallerCancelsMidEmit(t *testing.T) {
	ctx, cancel := context.WithCancelCause(context.Background())
	wantErr := errors.New("caller cancelled")

	produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
		// Fill the buffer so the next emit blocks on ctx.Done().
		for i := 0; i < DefaultEventBufferSize+2; i++ {
			if i == DefaultEventBufferSize {
				cancel(wantErr)
			}
			if err := emit(TextDeltaEvent{Delta: fmt.Sprintf("%d", i)}); err != nil {
				return nil, err
			}
		}
		return nil, nil
	}

	stream := Run(ctx, LLMRequest{Model: "test"}, produce)

	var got []LLMEvent
	for ev := range stream.Events {
		got = append(got, ev)
	}
	result := <-stream.Done

	// The cancellation and the buffered send race; we may see DefaultEventBufferSize
	// or DefaultEventBufferSize+1 events before emit returns the error.
	if len(got) < DefaultEventBufferSize || len(got) > DefaultEventBufferSize+1 {
		t.Fatalf("expected %d or %d events before cancellation blocked, got %d", DefaultEventBufferSize, DefaultEventBufferSize+1, len(got))
	}
	if result.Err == nil {
		t.Fatal("expected error after cancellation")
	}
	if !errors.Is(result.Err, wantErr) {
		t.Fatalf("expected %v, got %v", wantErr, result.Err)
	}
}

func TestRunCallerCancelsBeforeFirstEmit(t *testing.T) {
	ctx, cancel := context.WithCancelCause(context.Background())
	wantErr := errors.New("early cancel")
	cancel(wantErr)

	produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
		// Emit until the cancellation is observed.
		for i := 0; ; i++ {
			if err := emit(TextDeltaEvent{Delta: fmt.Sprintf("%d", i)}); err != nil {
				return nil, err
			}
		}
	}

	stream := Run(ctx, LLMRequest{Model: "test"}, produce)

	var got []LLMEvent
	for ev := range stream.Events {
		got = append(got, ev)
	}
	result := <-stream.Done

	// At most DefaultEventBufferSize events can be buffered before emit blocks
	// and observes the cancelled context.
	if len(got) > DefaultEventBufferSize {
		t.Fatalf("expected at most %d events, got %d", DefaultEventBufferSize, len(got))
	}
	if result.Err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(result.Err, wantErr) {
		t.Fatalf("expected %v, got %v", wantErr, result.Err)
	}
}

func TestRunSlowConsumer(t *testing.T) {
	req := LLMRequest{Model: "test"}

	produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
		for i := 0; i < DefaultEventBufferSize+5; i++ {
			if err := emit(TextDeltaEvent{Delta: fmt.Sprintf("%d", i)}); err != nil {
				return nil, err
			}
		}
		return nil, nil
	}

	stream := Run(context.Background(), req, produce)

	// Let the producer fill the buffer and block
	time.Sleep(50 * time.Millisecond)

	var got []LLMEvent
	for ev := range stream.Events {
		got = append(got, ev)
		time.Sleep(5 * time.Millisecond) // slow drain
	}
	result := <-stream.Done

	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
	if len(got) != DefaultEventBufferSize+5 {
		t.Fatalf("expected %d events, got %d", DefaultEventBufferSize+5, len(got))
	}
}

func TestRunAlreadyCancelled(t *testing.T) {
	ctx, cancel := context.WithCancelCause(context.Background())
	wantErr := errors.New("already done")
	cancel(wantErr)

	produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
		return nil, nil
	}

	stream := Run(ctx, LLMRequest{Model: "test"}, produce)

	// Should resolve promptly without leaking a goroutine
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range stream.Events {
		}
		<-stream.Done
	}()

	select {
	case <-done:
		// ok
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for already-cancelled stream")
	}
}

func TestRunConcurrentProducers(t *testing.T) {
	// Verify the runner is safe to invoke from multiple goroutines with different producers.
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
				return []Message{{Role: "assistant", Content: TextContent{Text: fmt.Sprintf("result-%d", n)}}}, nil
			}
			stream := Run(context.Background(), LLMRequest{Model: "test"}, produce)
			<-stream.Done
		}(i)
	}
	wg.Wait()
}
