package openaicompat

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

// scriptedStatusServer answers request n (0-based) with statuses[n], or the
// last status for later requests; 200 means a full successful SSE body.
func scriptedStatusServer(t *testing.T, headers []map[string]string, statuses ...int) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	calls := &atomic.Int32{}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := int(calls.Add(1)) - 1
		status := statuses[min(n, len(statuses)-1)]
		if n < len(headers) {
			for k, v := range headers[n] {
				w.Header().Set(k, v)
			}
		}
		if status != http.StatusOK {
			w.WriteHeader(status)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	t.Cleanup(ts.Close)
	return ts, calls
}

// A transient open failure is retried per policy: the second attempt succeeds,
// the ladder emits one LLMRetryEvent with the server hint, and GetAPIKey
// re-resolves per attempt so expiring credentials refresh across retries.
func TestProviderRetries429ThenSucceeds(t *testing.T) {
	ts, calls := scriptedStatusServer(t,
		[]map[string]string{{"retry-after-ms": "1"}},
		http.StatusTooManyRequests, http.StatusOK)

	var keyCalls atomic.Int32
	p, err := New(Config{
		Name:    "openai-compat",
		BaseURL: ts.URL,
		GetAPIKey: func(context.Context) (string, error) {
			keyCalls.Add(1)
			return "k", nil
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, result := streamAndCollect(t, p)
	if result.Err != nil {
		t.Fatalf("result.Err = %v, want nil (retry succeeded)", result.Err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("server calls = %d, want 2", got)
	}
	if got := keyCalls.Load(); got != 2 {
		t.Errorf("GetAPIKey calls = %d, want 2 (re-resolved per attempt)", got)
	}

	var retries []llm.LLMRetryEvent
	for _, ev := range events {
		if re, ok := ev.(llm.LLMRetryEvent); ok {
			retries = append(retries, re)
		}
		if e, ok := ev.(llm.LLMErrorEvent); ok {
			t.Errorf("unexpected LLMErrorEvent mid-ladder: %v", e.Error)
		}
	}
	if len(retries) != 1 {
		t.Fatalf("LLMRetryEvents = %d, want 1", len(retries))
	}
	re := retries[0]
	if re.Attempt != 1 || !re.ServerHint || re.NextDelay <= 0 {
		t.Errorf("LLMRetryEvent = %+v, want {Attempt: 1, ServerHint: true, NextDelay > 0}", re)
	}
	if re.Provider != "openai-compat" || re.Model != "test-model" {
		t.Errorf("LLMRetryEvent = {Provider: %q, Model: %q}, want {openai-compat, test-model}", re.Provider, re.Model)
	}
}

// Exhausted retries surface as the last transient failure: Transient on the
// LLMErrorEvent, and StreamResult.Err still unwraps as TransientError.
func TestProviderRetryExhausts(t *testing.T) {
	ts, calls := scriptedStatusServer(t,
		[]map[string]string{{"retry-after-ms": "1"}},
		http.StatusTooManyRequests)

	p, err := New(Config{
		Name:    "openai-compat",
		BaseURL: ts.URL,
		Retry:   llm.RetryPolicy{MaxRetries: 1},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, result := streamAndCollect(t, p)

	if got := calls.Load(); got != 2 {
		t.Errorf("server calls = %d, want 2 (initial + 1 retry)", got)
	}
	var terr *llm.TransientError
	if !errors.As(result.Err, &terr) {
		t.Errorf("result.Err = %v (%T), want errors.As *TransientError", result.Err, result.Err)
	}
	var llmErr *llm.LLMErrorEvent
	for _, ev := range events {
		if e, ok := ev.(llm.LLMErrorEvent); ok {
			llmErr = &e
		}
	}
	if llmErr == nil {
		t.Fatal("no LLMErrorEvent emitted")
	}
	if !llmErr.Transient {
		t.Error("LLMErrorEvent.Transient = false, want true (transient retries exhausted)")
	}
}

// x-should-retry: false overrides the status-based classification.
func TestProviderXShouldRetryFalseSkipsLadder(t *testing.T) {
	ts, calls := scriptedStatusServer(t,
		[]map[string]string{{"x-should-retry": "false"}},
		http.StatusServiceUnavailable)

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	events, result := streamAndCollect(t, p)

	if got := calls.Load(); got != 1 {
		t.Errorf("server calls = %d, want 1 (header veto)", got)
	}
	var terr *llm.TransientError
	if errors.As(result.Err, &terr) {
		t.Errorf("result.Err = %v, want a plain fatal (no TransientError)", result.Err)
	}
	for _, ev := range events {
		if _, ok := ev.(llm.LLMRetryEvent); ok {
			t.Error("LLMRetryEvent emitted despite x-should-retry: false")
		}
	}
}

// x-should-retry: true forces a retry even for a non-retryable status.
func TestProviderXShouldRetryTrueForcesLadder(t *testing.T) {
	ts, calls := scriptedStatusServer(t,
		[]map[string]string{{"x-should-retry": "true"}},
		http.StatusBadRequest, http.StatusOK)

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, result := streamAndCollect(t, p)

	if result.Err != nil {
		t.Fatalf("result.Err = %v, want nil (retry succeeded)", result.Err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("server calls = %d, want 2 (header forced the retry)", got)
	}
}

// Retries are disabled with MaxRetries < 0: one attempt, transient failure.
func TestProviderRetryDisabled(t *testing.T) {
	ts, calls := scriptedStatusServer(t, nil, http.StatusServiceUnavailable)

	p, err := New(Config{
		Name:    "openai-compat",
		BaseURL: ts.URL,
		Retry:   llm.RetryPolicy{MaxRetries: -1},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, result := streamAndCollect(t, p)

	if got := calls.Load(); got != 1 {
		t.Errorf("server calls = %d, want 1 (retries disabled)", got)
	}
	var terr *llm.TransientError
	if !errors.As(result.Err, &terr) {
		t.Errorf("result.Err = %v, want errors.As *TransientError", result.Err)
	}
}

// Fatal statuses (4xx outside 408/409/429) are not retried.
func TestProviderFatal4xxNotRetried(t *testing.T) {
	ts, calls := scriptedStatusServer(t, nil, http.StatusUnauthorized, http.StatusOK)

	p, err := New(Config{Name: "openai-compat", BaseURL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, result := streamAndCollect(t, p)

	if got := calls.Load(); got != 1 {
		t.Errorf("server calls = %d, want 1 (401 is fatal)", got)
	}
	var terr *llm.TransientError
	if errors.As(result.Err, &terr) {
		t.Errorf("result.Err = %v, want plain fatal for 401", result.Err)
	}
}
