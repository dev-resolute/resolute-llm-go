package llm

import (
	"context"
	"errors"
	"testing"
)

func TestRunOnBeforeRequestFiresOnce(t *testing.T) {
	var count int
	req := LLMRequest{
		Model: "test",
		OnBeforeRequest: func(headers map[string]string) error {
			count++
			return nil
		},
	}

	produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
		return nil, nil
	}

	stream := Run(context.Background(), req, produce)
	<-stream.Done

	if count != 1 {
		t.Fatalf("expected OnBeforeRequest to fire once, got %d", count)
	}
}

func TestRunOnBeforeRequestReceivesMergedHeaders(t *testing.T) {
	var gotHeaders map[string]string
	req := LLMRequest{
		Model:   "test",
		Headers: map[string]string{"X-B": "2"},
		OnBeforeRequest: func(headers map[string]string) error {
			gotHeaders = headers
			return nil
		},
	}

	produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
		return nil, nil
	}

	stream := Run(context.Background(), req, produce)
	<-stream.Done

	if gotHeaders == nil {
		t.Fatal("expected OnBeforeRequest to receive headers")
	}
	if gotHeaders["X-B"] != "2" {
		t.Fatalf("expected X-B=2, got %q", gotHeaders["X-B"])
	}
}

func TestRunOnBeforeRequestMutationReachesProducer(t *testing.T) {
	req := LLMRequest{
		Model: "test",
		OnBeforeRequest: func(headers map[string]string) error {
			headers["X-C"] = "3"
			return nil
		},
	}

	var producerHeaders map[string]string
	produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
		producerHeaders = headers
		return nil, nil
	}

	stream := Run(context.Background(), req, produce)
	<-stream.Done

	if producerHeaders["X-C"] != "3" {
		t.Fatalf("expected producer to see X-C=3, got %q", producerHeaders["X-C"])
	}
}

func TestRunOnBeforeRequestErrorAborts(t *testing.T) {
	wantErr := errors.New("aborted")
	req := LLMRequest{
		Model: "test",
		OnBeforeRequest: func(headers map[string]string) error {
			return wantErr
		},
	}

	var producerCalled bool
	produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
		producerCalled = true
		return nil, nil
	}

	stream := Run(context.Background(), req, produce)
	result := <-stream.Done

	if producerCalled {
		t.Fatal("expected producer to NOT be called when OnBeforeRequest errors")
	}
	if result.Err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(result.Err, wantErr) {
		t.Fatalf("expected %v, got %v", wantErr, result.Err)
	}
}

func TestRunOnAfterResponseFiresOnce(t *testing.T) {
	var count int
	req := LLMRequest{
		Model: "test",
		OnAfterResponse: func(statusCode int, headers map[string]string) {
			count++
		},
	}

	produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
		if setResponseMeta != nil {
			setResponseMeta(200, map[string]string{"X-Request-ID": "abc"})
		}
		return nil, nil
	}

	stream := Run(context.Background(), req, produce)
	<-stream.Done

	if count != 1 {
		t.Fatalf("expected OnAfterResponse to fire once, got %d", count)
	}
}

func TestRunOnAfterResponseReceivesStatusAndHeaders(t *testing.T) {
	var gotStatus int
	var gotHeaders map[string]string
	req := LLMRequest{
		Model: "test",
		OnAfterResponse: func(statusCode int, headers map[string]string) {
			gotStatus = statusCode
			gotHeaders = headers
		},
	}

	produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
		if setResponseMeta != nil {
			setResponseMeta(200, map[string]string{"X-Request-ID": "req-123"})
		}
		return nil, nil
	}

	stream := Run(context.Background(), req, produce)
	<-stream.Done

	if gotStatus != 200 {
		t.Fatalf("expected status 200, got %d", gotStatus)
	}
	if gotHeaders["X-Request-ID"] != "req-123" {
		t.Fatalf("expected X-Request-ID=req-123, got %q", gotHeaders["X-Request-ID"])
	}
}

func TestRunOnAfterResponseFiresOnProducerError(t *testing.T) {
	var count int
	wantErr := errors.New("producer failed")
	req := LLMRequest{
		Model: "test",
		OnAfterResponse: func(statusCode int, headers map[string]string) {
			count++
		},
	}

	produce := func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error) {
		if setResponseMeta != nil {
			setResponseMeta(503, map[string]string{})
		}
		return nil, wantErr
	}

	stream := Run(context.Background(), req, produce)
	result := <-stream.Done

	if count != 1 {
		t.Fatalf("expected OnAfterResponse to fire once on error, got %d", count)
	}
	if result.Err == nil {
		t.Fatal("expected producer error")
	}
}
