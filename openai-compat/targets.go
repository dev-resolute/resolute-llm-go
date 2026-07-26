package openaicompat

import (
	"context"

	"github.com/dev-resolute/resolute-llm-go"
)

// TargetConfig carries the app-owned settings the built-in provider constructors
// (XAI, Mistral, Qwen, ZAI) accept. Each constructor supplies the Name, BaseURL,
// thinking dialect, and capability classifier itself; the app supplies only the
// credentials and any extra headers. Source the key from the environment via
// GetAPIKey (or APIKey), never a literal.
type TargetConfig struct {
	APIKey    string
	GetAPIKey func(ctx context.Context) (string, error)
	Headers   map[string]string
}

// newTarget builds a Provider for a built-in family: the constructor fixes Name,
// BaseURL, Compat, and the classifier; the caller's TargetConfig supplies auth.
func newTarget(cfg TargetConfig, name, baseURL string, compat Compat, classify func(string) classification) (llm.LLMProvider, error) {
	p, err := newProvider(Config{
		Name:      name,
		BaseURL:   baseURL,
		APIKey:    cfg.APIKey,
		GetAPIKey: cfg.GetAPIKey,
		Headers:   cfg.Headers,
		Compat:    compat,
	})
	if err != nil {
		return nil, err
	}
	p.classify = classify
	return p, nil
}

// classification is the per-model behaviour a provider-family classifier resolves
// from a bare model id, catalog-free (ADR-0008). It mirrors what classifyGemini
// does for the native Gemini provider: drive ProviderCapabilities and, for the
// reasoning_effort dialect, gate whether the param is sent.
type classification struct {
	// thinking reports whether the model reasons at all.
	thinking bool
	// vision reports whether the model accepts image input.
	vision bool
	// reasoningEffort reports whether the model accepts the reasoning_effort param.
	// Some models reason but reject it (xAI grok-4 / Mistral Magistral always-reason),
	// so this is distinct from thinking and only consulted by the default dialect.
	reasoningEffort bool
	// strictTools reports whether this model may receive provider-side "strict"
	// JSON-schema-enforced tool sampling. Every family we ship defaults to true;
	// upstream denylists only moonshot/together/cloudflare-gateway/nvidia
	// (openai-completions.ts:1454), none of which are current named families —
	// this field is where a future denylisted family would flip the switch off.
	strictTools bool
}
