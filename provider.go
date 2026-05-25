// Package llm provides a provider-agnostic abstraction over LLM streaming APIs.
package llm

import "context"

// LLMProvider is the single interface implemented by every concrete provider.
// It abstracts differences between LLM wire protocols without imposing agent-loop opinions.
type LLMProvider interface {
	// Name returns the provider's short identifier, used in model references
	// like "<name>/<model-id>".
	Name() string

	// Capabilities returns the feature set for the given model.
	Capabilities(model string) ProviderCapabilities

	// Stream initiates a streaming LLM call. The returned EventStream delivers
	// typed events on Events and a single terminal result on Done.
	// Cancellation is honored via ctx; the Events channel closes when the
	// stream completes or ctx is cancelled.
	Stream(ctx context.Context, req LLMRequest) EventStream
}
