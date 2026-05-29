package llm

// LLMRequest carries all inputs for a single streaming LLM call.
type LLMRequest struct {
	Model    string
	Messages []Message
	Tools    []ToolDef
	Thinking ThinkingLevel
	// SessionID optionally identifies the conversation this call belongs to.
	// The OpenAI-compatible adapter sends it as prompt-cache affinity headers
	// (session_id, x-client-request-id, x-session-affinity) and as the
	// prompt_cache_key body param, so repeated calls route to the same replica
	// and maximize prompt-cache hits. The Gemini adapter ignores it. Empty means
	// "no session hint".
	SessionID string
	// ThinkingBudgets optionally overrides the per-level token budget for the
	// active Thinking level. Providers whose reasoning control is token-based
	// (Gemini's thinking_budget) apply it; providers whose control is categorical
	// (OpenAI-compatible reasoning_effort) ignore it. Nil means "use provider
	// defaults". ProviderHints, when set, takes precedence over this map.
	ThinkingBudgets map[ThinkingLevel]int
	// Transport is the preferred stream transport. Providers that support only
	// HTTP/SSE honor TransportAuto and TransportSSE; TransportWebSocket returns
	// ErrTransportUnsupported until a websocket-capable provider exists.
	Transport       TransportPreference
	ProviderHints   ProviderHints
	Retry           RetryPolicy
	Headers         map[string]string
	OnBeforeRequest func(headers map[string]string) error
	OnAfterResponse func(statusCode int, headers map[string]string)
}
