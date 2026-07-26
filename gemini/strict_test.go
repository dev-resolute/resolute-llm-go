package gemini

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
	"google.golang.org/genai"
)

// strictPreferTool builds a tool opted into "prefer" strict JSON-schema
// sampling: use it when the model supports validated tool calling, silently
// fall back otherwise.
func strictPreferTool(name string) llm.ToolDef {
	return llm.ToolDef{
		Name:                name,
		Schema:              json.RawMessage(`{"type":"object"}`),
		ConstrainedSampling: &llm.ConstrainedSampling{Strict: llm.StrictPrefer},
	}
}

// TestStrictToolSamplingSupported is a direct table test over the support
// predicate: Gemini 3 pro/flash support validated function calling; Gemini
// 2.5, legacy models, and Gemma 4 do not (upstream supportsGoogleStrictToolSampling
// gates on ^gemini-, which never matches Gemma ids).
func TestStrictToolSamplingSupported(t *testing.T) {
	tests := []struct {
		name  string
		model string
		want  bool
	}{
		{name: "3 pro preview", model: "gemini-3.1-pro-preview", want: true},
		{name: "3 flash", model: "gemini-3.5-flash", want: true},
		{name: "2.5 flash unsupported", model: "gemini-2.5-flash", want: false},
		{name: "legacy unsupported", model: "gemini-1.5-pro", want: false},
		{name: "gemma 4 excluded", model: "gemma-4", want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := strictToolSamplingSupported(tt.model); got != tt.want {
				t.Errorf("strictToolSamplingSupported(%q) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

// TestStrictPreferSetsValidatedModeOnGemini3: a "prefer" strict tool on a
// Gemini 3 model turns on VALIDATED function-calling mode request-wide
// (upstream google-shared.ts:311-324 — the mode is request-level, not
// per-tool).
func TestStrictPreferSetsValidatedModeOnGemini3(t *testing.T) {
	// given a "prefer" strict tool on a Gemini 3 model
	req := llm.LLMRequest{
		Model: "gemini-3.1-pro-preview",
		Tools: []llm.ToolDef{strictPreferTool("lookup")},
	}

	// when the provider builds the genai config
	config, err := toGeminiConfig(req, nil)
	if err != nil {
		t.Fatalf("toGeminiConfig: unexpected error: %v", err)
	}

	// then ToolConfig carries VALIDATED function-calling mode
	if config.ToolConfig == nil || config.ToolConfig.FunctionCallingConfig == nil {
		t.Fatal("ToolConfig/FunctionCallingConfig is nil, want VALIDATED mode")
	}
	if got := config.ToolConfig.FunctionCallingConfig.Mode; got != genai.FunctionCallingConfigModeValidated {
		t.Errorf("FunctionCallingConfig.Mode = %q, want %q", got, genai.FunctionCallingConfigModeValidated)
	}
}

// TestStrictPreferFallsBackSilentlyOnGemini25: a "prefer" strict tool on a
// model that doesn't support validated tool calling falls back without error
// and without setting ToolConfig at all.
func TestStrictPreferFallsBackSilentlyOnGemini25(t *testing.T) {
	// given a "prefer" strict tool on a Gemini 2.5 model (unsupported)
	req := llm.LLMRequest{
		Model: "gemini-2.5-flash",
		Tools: []llm.ToolDef{strictPreferTool("lookup")},
	}

	// when the provider builds the genai config
	config, err := toGeminiConfig(req, nil)
	if err != nil {
		t.Fatalf("toGeminiConfig: unexpected error: %v", err)
	}

	// then no ToolConfig is set: silent fallback, not an error
	if config.ToolConfig != nil {
		t.Errorf("ToolConfig = %+v, want nil (silent fallback on unsupported model)", config.ToolConfig)
	}
}

// TestStrictRequireUnsupportedFatal: a "require" strict tool on a model that
// doesn't support validated tool calling fails before the request is ever
// issued, with the exact upstream message and errors.Is(llm.ErrProviderFatal)
// so retry ladders stop retrying a deterministic config error (LLM-11 polarity).
func TestStrictRequireUnsupportedFatal(t *testing.T) {
	// given a "require" strict tool on a Gemini 2.5 model (unsupported)
	req := llm.LLMRequest{
		Model: "gemini-2.5-flash",
		Tools: []llm.ToolDef{
			{
				Name:                "lookup",
				Schema:              json.RawMessage(`{"type":"object"}`),
				ConstrainedSampling: &llm.ConstrainedSampling{Strict: llm.StrictRequire},
			},
		},
	}

	// when the provider builds the genai config
	_, err := toGeminiConfig(req, nil)

	// then it fails with the exact upstream message, wrapped in ErrProviderFatal
	if err == nil {
		t.Fatal("toGeminiConfig: expected error, got nil")
	}
	if !errors.Is(err, llm.ErrProviderFatal) {
		t.Errorf("errors.Is(err, llm.ErrProviderFatal) = false, err = %v", err)
	}
	const wantMsg = `Tool "lookup" requires JSON-schema constrained sampling, but strict tools are unsupported.`
	if !strings.Contains(err.Error(), wantMsg) {
		t.Errorf("err = %q, want it to contain %q", err.Error(), wantMsg)
	}
}

// TestNoConstrainedToolsOmitsToolConfigOnGemini3: VALIDATED mode only appears
// when a tool actually resolves strict -- plain tools on a Gemini 3 model
// (which otherwise supports the mode) must not trigger it.
func TestNoConstrainedToolsOmitsToolConfigOnGemini3(t *testing.T) {
	// given tools with no ConstrainedSampling opt-in, on a Gemini 3 model
	req := llm.LLMRequest{
		Model: "gemini-3.1-pro-preview",
		Tools: []llm.ToolDef{
			{Name: "plain", Schema: json.RawMessage(`{"type":"object"}`)},
		},
	}

	// when the provider builds the genai config
	config, err := toGeminiConfig(req, nil)
	if err != nil {
		t.Fatalf("toGeminiConfig: unexpected error: %v", err)
	}

	// then no ToolConfig is set: mode only appears when strict engages
	if config.ToolConfig != nil {
		t.Errorf("ToolConfig = %+v, want nil (no tool opted into strict sampling)", config.ToolConfig)
	}
}
