package llm

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"time"
)

// RetryPolicy configures the provider retry ladder (Retry). The zero value
// resolves to the documented defaults (DefaultMaxRetries, DefaultMaxRetryDelay)
// — matching upstream's agent policy — and negative fields disable:
// MaxRetries < 0 runs the operation once; MaxRetryDelay < 0 lifts the cap on
// server-requested waits.
type RetryPolicy struct {
	// MaxRetries bounds retry attempts after the initial call. 0 resolves to
	// DefaultMaxRetries; negative disables retries.
	MaxRetries int
	// MaxRetryDelay caps a server-requested (retry-after) wait: a hint above
	// the cap fails immediately (upstream rule). 0 resolves to
	// DefaultMaxRetryDelay; negative disables the cap. The exponential backoff
	// is never capped.
	MaxRetryDelay time.Duration
}

// Defaults for RetryPolicy zero values.
const (
	DefaultMaxRetries    = 3
	DefaultMaxRetryDelay = 60 * time.Second
)

// resolve fills zero fields with the defaults and normalizes disable sentinels.
func (p RetryPolicy) resolve() RetryPolicy {
	if p.MaxRetries == 0 {
		p.MaxRetries = DefaultMaxRetries
	}
	if p.MaxRetries < 0 {
		p.MaxRetries = 0
	}
	if p.MaxRetryDelay == 0 {
		p.MaxRetryDelay = DefaultMaxRetryDelay
	}
	if p.MaxRetryDelay < 0 {
		p.MaxRetryDelay = 0 // no cap
	}
	return p
}

// backoff computes the exponential wait before retry attempt (0-based):
// min(0.5·2^attempt, 8)s with −0–25% jitter (upstream getRetryDelayMs).
func (p RetryPolicy) backoff(attempt int) time.Duration {
	base := 500 * time.Millisecond
	for ; attempt > 0 && base < 8*time.Second; attempt-- {
		base *= 2
	}
	if base > 8*time.Second {
		base = 8 * time.Second
	}
	return base - time.Duration(rand.Float64()*0.25*float64(base))
}

// TransientError wraps a stream-open failure the provider classifies as
// retryable, optionally carrying the server's requested wait (retry-after).
// Providers return it from their Retry op; Retry retries it per policy.
type TransientError struct {
	Err error
	// RetryAfter is the server-requested wait (retry-after / retry-after-ms
	// response headers). 0 when the server gave no hint — the ladder then uses
	// the exponential backoff.
	RetryAfter time.Duration
}

func (e *TransientError) Error() string { return e.Err.Error() }
func (e *TransientError) Unwrap() error { return e.Err }

// Retry runs op, retrying TransientError failures per policy (upstream
// retryProviderRequest). It emits an LLMRetryEvent before each retry wait.
//
// The retried boundary is the stream-open phase: op must be idempotent and
// must not emit events (a successful open is followed by unretried streaming,
// so content is never duplicated). Classification is the provider's: anything
// not wrapped in TransientError passes through unretried.
//
// Failure modes: a server-requested wait above policy.MaxRetryDelay fails
// immediately (naming both delays, upstream's message shape); exhausted
// retries return the last TransientError (so the provider still classifies
// the failure as transient); context cancellation stops the ladder with the
// ctx cause.
func Retry(ctx context.Context, policy RetryPolicy, provider, model string, emit func(LLMEvent) error, op func(ctx context.Context) error) error {
	p := policy.resolve()
	for attempt := 0; ; attempt++ {
		err := op(ctx)
		if err == nil {
			return nil
		}
		if ctx.Err() != nil {
			return context.Cause(ctx)
		}
		var terr *TransientError
		if !errors.As(err, &terr) {
			return err
		}
		if attempt >= p.MaxRetries {
			return err
		}

		var delay time.Duration
		if terr.RetryAfter > 0 {
			if p.MaxRetryDelay > 0 && terr.RetryAfter > p.MaxRetryDelay {
				return fmt.Errorf("%s: server requested %ds retry delay (max: %ds): %w",
					provider, ceilSeconds(terr.RetryAfter), ceilSeconds(p.MaxRetryDelay), terr.Err)
			}
			delay = terr.RetryAfter
		} else {
			delay = p.backoff(attempt)
		}

		if emitErr := emit(LLMRetryEvent{
			Provider:   provider,
			Model:      model,
			Attempt:    attempt + 1,
			NextDelay:  delay,
			Reason:     terr.Err.Error(),
			ServerHint: terr.RetryAfter > 0,
		}); emitErr != nil {
			return emitErr
		}
		if sleepErr := sleep(ctx, delay); sleepErr != nil {
			return sleepErr
		}
	}
}

// sleep waits d or until ctx is done, returning the ctx cause in the latter
// case (upstream abortableSleep).
func sleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return context.Cause(ctx)
	case <-t.C:
		return nil
	}
}

func ceilSeconds(d time.Duration) int64 {
	return int64(d/time.Second) + b2i64(d%time.Second != 0)
}

func b2i64(b bool) int64 {
	if b {
		return 1
	}
	return 0
}
