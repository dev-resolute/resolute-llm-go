package llm

import "errors"

// Sentinel errors for pi-llm-go.
var (
	ErrProviderFatal        = errors.New("provider fatal error")
	ErrInvalidModel         = errors.New("invalid model")
	ErrUnsupportedFeature   = errors.New("unsupported feature")
	ErrMalformedResponse    = errors.New("malformed provider response")
	ErrTransportUnsupported = errors.New("transport not supported by provider")
)
