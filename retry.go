package llm

import "time"

// RetryPolicy configures retry behavior for a provider call.
// The actual retry logic is delegated to the underlying SDK; this struct is the
// configuration shape shared across providers.
type RetryPolicy struct {
	MaxRetries    int
	MaxRetryDelay time.Duration
}

// Defaults for RetryPolicy zero values.
const (
	DefaultMaxRetries    = 3
	DefaultMaxRetryDelay = 60 * time.Second
)
