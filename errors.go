package llm

import "errors"

// Sentinel errors for pi-llm-go.
var (
	ErrProviderFatal        = errors.New("provider fatal error")
	ErrInvalidModel         = errors.New("invalid model")
	ErrUnsupportedFeature   = errors.New("unsupported feature")
	ErrMalformedResponse    = errors.New("malformed provider response")
	ErrTransportUnsupported = errors.New("transport not supported by provider")
	// ErrProviderStop marks a provider-terminated message: the stream ended
	// with a terminal stop/finish reason that has no portable mapping (Gemini
	// SAFETY/RECITATION/..., OpenAI content_filter/network_error, or a
	// genuinely unknown reason). Fatal — the message did not complete.
	ErrProviderStop    = errors.New("provider stop")
	ErrContextOverflow = errors.New("context length exceeded")
)
