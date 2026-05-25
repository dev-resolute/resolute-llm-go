package llm

import (
	"context"
	"fmt"
	"sync"
)

// DefaultEventBufferSize is the default buffer size for EventStream.Events.
const DefaultEventBufferSize = 16

// ProduceFn is the callback signature for provider-specific streaming logic.
// The producer receives an emit function that handles ctx cancellation.
// It returns the new messages produced during the stream and any error.
// The headers map is the merged result of Config.Headers + LLMRequest.Headers +
// any mutations made by OnBeforeRequest hooks.
// setResponseMeta lets the producer report HTTP status + response headers for
// the OnAfterResponse hook. It may be called at most once.
type ProduceFn func(ctx context.Context, req LLMRequest, emit func(LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]Message, error)

// Run executes a streaming LLM call using the given producer callback.
// It allocates channels, manages goroutine lifetime, and enforces the
// EventStream contract: Events closes when the stream ends, Done delivers
// exactly one StreamResult with Messages = req.Messages ++ produced.
func Run(ctx context.Context, req LLMRequest, produce ProduceFn) EventStream {
	evCh := make(chan LLMEvent, DefaultEventBufferSize)
	doneCh := make(chan StreamResult, 1)

	go func() {
		defer close(evCh)
		defer close(doneCh)

		emit := func(ev LLMEvent) error {
			select {
			case evCh <- ev:
				return nil
			case <-ctx.Done():
				return context.Cause(ctx)
			}
		}

		mergedHeaders := make(map[string]string)
		// Config headers are already available to the provider; the runner
		// only needs to surface the per-request + hook-mutated headers.
		for k, v := range req.Headers {
			mergedHeaders[k] = v
		}

		if req.OnBeforeRequest != nil {
			if err := req.OnBeforeRequest(mergedHeaders); err != nil {
				doneCh <- StreamResult{Err: fmt.Errorf("before provider request hook: %w", err)}
				return
			}
		}

		// Provider captures status/headers for AfterProviderResponse.
		var observedStatus int
		var observedHeaders map[string]string
		var metaOnce sync.Once
		setResponseMeta := func(status int, respHeaders map[string]string) {
			metaOnce.Do(func() {
				observedStatus = status
				observedHeaders = respHeaders
			})
		}

		msgs, err := produce(ctx, req, emit, mergedHeaders, setResponseMeta)

		if req.OnAfterResponse != nil {
			req.OnAfterResponse(observedStatus, observedHeaders)
		}

		doneCh <- StreamResult{Messages: append(req.Messages, msgs...), Err: err}
	}()

	return NewEventStream(evCh, doneCh)
}
