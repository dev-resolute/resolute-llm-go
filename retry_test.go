package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestRetryPolicyResolve(t *testing.T) {
	for _, tt := range []struct {
		name string
		in   RetryPolicy
		want RetryPolicy
	}{
		{"zero value defaults", RetryPolicy{}, RetryPolicy{MaxRetries: DefaultMaxRetries, MaxRetryDelay: DefaultMaxRetryDelay}},
		{"explicit values kept", RetryPolicy{MaxRetries: 5, MaxRetryDelay: time.Second}, RetryPolicy{MaxRetries: 5, MaxRetryDelay: time.Second}},
		{"retries disabled", RetryPolicy{MaxRetries: -1}, RetryPolicy{MaxRetries: 0, MaxRetryDelay: DefaultMaxRetryDelay}},
		{"cap disabled", RetryPolicy{MaxRetryDelay: -1}, RetryPolicy{MaxRetries: DefaultMaxRetries, MaxRetryDelay: 0}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.in.resolve(); got != tt.want {
				t.Errorf("RetryPolicy%+v.resolve() = %+v, want %+v", tt.in, got, tt.want)
			}
		})
	}
}

func TestBackoff(t *testing.T) {
	for attempt, wantBase := range []time.Duration{
		500 * time.Millisecond, time.Second, 2 * time.Second, 4 * time.Second,
		8 * time.Second, 8 * time.Second, 8 * time.Second, // capped
	} {
		for range 20 {
			got := RetryPolicy{}.backoff(attempt)
			// jitter subtracts 0–25%: result lands in [0.75·base, base]
			if got > wantBase || got < wantBase*3/4 {
				t.Fatalf("backoff(%d) = %v, want within [0.75, 1] × %v", attempt, got, wantBase)
			}
		}
	}
}

// scriptedOp returns an op that fails with each of errs in turn, then succeeds.
func scriptedOp(errs ...error) (func(context.Context) error, *int) {
	calls := new(int)
	return func(context.Context) error {
		*calls++
		if *calls <= len(errs) {
			return errs[*calls-1]
		}
		return nil
	}, calls
}

func collectEvents(ev LLMEvent) error { return nil }

func TestRetrySucceedsFirstTry(t *testing.T) {
	op, calls := scriptedOp()
	if err := Retry(context.Background(), RetryPolicy{}, "p", "m", collectEvents, op); err != nil {
		t.Fatalf("Retry: %v", err)
	}
	if *calls != 1 {
		t.Errorf("op calls = %d, want 1", *calls)
	}
}

func TestRetryRetriesTransientThenSucceeds(t *testing.T) {
	transient := &TransientError{Err: errors.New("boom"), RetryAfter: time.Millisecond}
	op, calls := scriptedOp(transient, transient)

	var events []LLMRetryEvent
	emit := func(ev LLMEvent) error {
		if re, ok := ev.(LLMRetryEvent); ok {
			events = append(events, re)
		}
		return nil
	}

	if err := Retry(context.Background(), RetryPolicy{}, "p", "m", emit, op); err != nil {
		t.Fatalf("Retry: %v", err)
	}
	if *calls != 3 {
		t.Errorf("op calls = %d, want 3 (2 failures + success)", *calls)
	}
	if len(events) != 2 {
		t.Fatalf("LLMRetryEvents = %d, want 2", len(events))
	}
	for i, ev := range events {
		if ev.Attempt != i+1 {
			t.Errorf("event %d Attempt = %d, want %d (1-based)", i, ev.Attempt, i+1)
		}
		if ev.Provider != "p" || ev.Model != "m" {
			t.Errorf("event %d = {Provider: %q, Model: %q}, want {p, m}", i, ev.Provider, ev.Model)
		}
		if !ev.ServerHint || ev.NextDelay != time.Millisecond {
			t.Errorf("event %d = {ServerHint: %v, NextDelay: %v}, want {true, 1ms}", i, ev.ServerHint, ev.NextDelay)
		}
	}
}

func TestRetryExhausts(t *testing.T) {
	transient := &TransientError{Err: errors.New("boom"), RetryAfter: time.Millisecond}
	op, calls := scriptedOp(transient, transient, transient)

	err := Retry(context.Background(), RetryPolicy{MaxRetries: 1}, "p", "m", collectEvents, op)
	if err == nil {
		t.Fatal("Retry err = nil, want the exhausted transient error")
	}
	var terr *TransientError
	if !errors.As(err, &terr) {
		t.Errorf("exhausted error is %T, want *TransientError so providers mark it transient", err)
	}
	if *calls != 2 {
		t.Errorf("op calls = %d, want 2 (initial + 1 retry)", *calls)
	}
}

func TestRetryFatalPassesThroughUnretried(t *testing.T) {
	fatal := fmt.Errorf("wrap: %w", ErrProviderFatal)
	op, calls := scriptedOp(fatal)

	err := Retry(context.Background(), RetryPolicy{}, "p", "m", collectEvents, op)
	if !errors.Is(err, ErrProviderFatal) {
		t.Errorf("Retry err = %v, want errors.Is ErrProviderFatal", err)
	}
	if *calls != 1 {
		t.Errorf("op calls = %d, want 1 (fatal errors are not retried)", *calls)
	}
}

func TestRetryDisabledRunsOnce(t *testing.T) {
	transient := &TransientError{Err: errors.New("boom")}
	op, calls := scriptedOp(transient)

	err := Retry(context.Background(), RetryPolicy{MaxRetries: -1}, "p", "m", collectEvents, op)
	if err == nil {
		t.Fatal("Retry err = nil, want the transient error (no retries configured)")
	}
	if *calls != 1 {
		t.Errorf("op calls = %d, want 1 (retries disabled)", *calls)
	}
}

func TestRetryOverCapServerHintFailsImmediately(t *testing.T) {
	transient := &TransientError{Err: errors.New("slow down"), RetryAfter: 120 * time.Second}
	op, calls := scriptedOp(transient)

	err := Retry(context.Background(), RetryPolicy{}, "p", "m", collectEvents, op)
	if err == nil {
		t.Fatal("Retry err = nil, want the over-cap failure")
	}
	// upstream message shape: names both delays and the provider error
	want := "server requested 120s retry delay (max: 60s)"
	if msg := err.Error(); !strings.Contains(msg, want) || !strings.Contains(msg, "slow down") {
		t.Errorf("err = %q, want it to contain %q and the provider message", msg, want)
	}
	if *calls != 1 {
		t.Errorf("op calls = %d, want 1 (over-cap hint fails without retrying)", *calls)
	}
}

func TestRetryUncappedServerHintHonored(t *testing.T) {
	transient := &TransientError{Err: errors.New("boom"), RetryAfter: 2 * time.Millisecond}
	op, calls := scriptedOp(transient)

	var events []LLMRetryEvent
	emit := func(ev LLMEvent) error {
		if re, ok := ev.(LLMRetryEvent); ok {
			events = append(events, re)
		}
		return nil
	}

	if err := Retry(context.Background(), RetryPolicy{MaxRetryDelay: -1}, "p", "m", emit, op); err != nil {
		t.Fatalf("Retry: %v", err)
	}
	if *calls != 2 {
		t.Errorf("op calls = %d, want 2", *calls)
	}
	if len(events) != 1 || events[0].NextDelay != 2*time.Millisecond {
		t.Errorf("events = %+v, want one event with NextDelay 2ms (hint honored, cap disabled)", events)
	}
}

func TestRetryBackoffUsedWithoutHint(t *testing.T) {
	// no hint: the ladder waits the exponential backoff (attempt 0 = 500ms ± jitter)
	transient := &TransientError{Err: errors.New("boom")}
	op, calls := scriptedOp(transient)

	var events []LLMRetryEvent
	emit := func(ev LLMEvent) error {
		if re, ok := ev.(LLMRetryEvent); ok {
			events = append(events, re)
		}
		return nil
	}

	start := time.Now()
	if err := Retry(context.Background(), RetryPolicy{}, "p", "m", emit, op); err != nil {
		t.Fatalf("Retry: %v", err)
	}
	elapsed := time.Since(start)
	if *calls != 2 {
		t.Errorf("op calls = %d, want 2", *calls)
	}
	if len(events) != 1 || events[0].ServerHint {
		t.Fatalf("events = %+v, want one hintless event", events)
	}
	if d := events[0].NextDelay; d > 500*time.Millisecond || d < 375*time.Millisecond {
		t.Errorf("NextDelay = %v, want within [0.75, 1] × 500ms", d)
	}
	if elapsed < 300*time.Millisecond {
		t.Errorf("ladder returned after %v, want it to actually wait the backoff", elapsed)
	}
}

func TestRetryContextCancelDuringWait(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	transient := &TransientError{Err: errors.New("boom"), RetryAfter: 10 * time.Second}
	op, calls := scriptedOp(transient)

	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	err := Retry(ctx, RetryPolicy{}, "p", "m", collectEvents, op)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Retry err = %v, want context.Canceled", err)
	}
	if *calls != 1 {
		t.Errorf("op calls = %d, want 1 (cancel stops the ladder, no second attempt)", *calls)
	}
}

func TestTransientErrorUnwraps(t *testing.T) {
	base := errors.New("429 too many requests")
	terr := &TransientError{Err: base, RetryAfter: time.Second}
	if !errors.Is(terr, base) {
		t.Error("errors.Is(TransientError, base) = false, want true (Unwrap)")
	}
	if terr.Error() != base.Error() {
		t.Errorf("Error() = %q, want %q", terr.Error(), base.Error())
	}
}
