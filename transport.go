package llm

// TransportPreference is a portable preference for the stream transport a
// provider uses. Providers that support only one transport honor TransportAuto
// and TransportSSE by using HTTP/SSE; a provider that does not implement the
// requested transport returns ErrTransportUnsupported rather than silently
// falling back, so callers learn the transport is unavailable.
type TransportPreference int

const (
	TransportAuto TransportPreference = iota
	TransportSSE
	TransportWebSocket
)

// String returns the wire-style name of the transport preference.
func (t TransportPreference) String() string {
	switch t {
	case TransportAuto:
		return "auto"
	case TransportSSE:
		return "sse"
	case TransportWebSocket:
		return "websocket"
	default:
		return "unknown"
	}
}
