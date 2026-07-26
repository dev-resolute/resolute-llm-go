package llm

import (
	"encoding/json"
	"fmt"
)

// StrictMode selects how strongly a tool requests JSON-schema-enforced calls.
type StrictMode string

const (
	StrictPrefer  StrictMode = "prefer"  // use when supported, silently fall back
	StrictRequire StrictMode = "require" // error when the provider can't honor it
)

// ConstrainedSampling opts a tool into provider-side constrained sampling.
// Nil disables it (upstream's omitted/false).
type ConstrainedSampling struct {
	Strict StrictMode
}

// ToolDef is the LLM-visible tool specification.
type ToolDef struct {
	Name                string
	Description         string
	Schema              json.RawMessage
	ConstrainedSampling *ConstrainedSampling
}

// ResolveStrictSampling reports whether tool should be sent with provider-side
// strict JSON-schema enforcement. Ports upstream
// resolveJsonSchemaStrictSampling (pi v0.82.0 constrained-sampling.ts):
// nil config → (false, nil); prefer+supported → (true, nil);
// prefer+unsupported → (false, nil); require+unsupported → error with the
// upstream message. A non-nil config with Strict outside {prefer, require}
// (including "") errors naming the invalid value.
func ResolveStrictSampling(tool ToolDef, supported bool) (bool, error) {
	cfg := tool.ConstrainedSampling
	if cfg == nil {
		return false, nil
	}
	switch cfg.Strict {
	case StrictPrefer, StrictRequire:
	default:
		return false, fmt.Errorf("tool %q has invalid ConstrainedSampling.Strict %q (want %q or %q)", tool.Name, cfg.Strict, StrictPrefer, StrictRequire)
	}
	if supported {
		return true, nil
	}
	if cfg.Strict == StrictRequire {
		return false, fmt.Errorf("Tool %q requires JSON-schema constrained sampling, but strict tools are unsupported.", tool.Name)
	}
	return false, nil
}
