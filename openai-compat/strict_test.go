package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

// toolFunction extracts the {"type":"function","function":{...}} object for
// the named tool from a captured request body's "tools" array.
func toolFunction(t *testing.T, captured map[string]any, name string) map[string]any {
	t.Helper()
	rawTools, ok := captured["tools"].([]any)
	if !ok {
		t.Fatalf("captured tools has type %T, want []any", captured["tools"])
	}
	for _, raw := range rawTools {
		tool, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		fn, ok := tool["function"].(map[string]any)
		if !ok {
			continue
		}
		if fn["name"] == name {
			return fn
		}
	}
	t.Fatalf("tool %q not found in captured tools: %+v", name, rawTools)
	return nil
}

// TestStrictEmissionSupported: a default instance (no SupportsStrictTools
// override, no classifier) supports strict tools. Every function tool object
// carries "strict": true when the tool resolves strict, false otherwise —
// upstream convertTools parity.
func TestStrictEmissionSupported(t *testing.T) {
	req := llm.LLMRequest{
		Model: "test-model",
		Tools: []llm.ToolDef{
			{
				Name:                "preferred",
				Description:         "uses strict when available",
				Schema:              json.RawMessage(`{"type":"object"}`),
				ConstrainedSampling: &llm.ConstrainedSampling{Strict: llm.StrictPrefer},
			},
			{
				Name:        "plain",
				Description: "no constrained sampling opt-in",
				Schema:      json.RawMessage(`{"type":"object"}`),
			},
		},
	}

	captured := captureBody(t, Config{}, req)

	preferred := toolFunction(t, captured, "preferred")
	if strict, ok := preferred["strict"].(bool); !ok || !strict {
		t.Errorf(`"preferred" tool strict = %v (ok=%v), want true`, preferred["strict"], ok)
	}

	plain := toolFunction(t, captured, "plain")
	if strict, ok := plain["strict"].(bool); !ok || strict {
		t.Errorf(`"plain" tool strict = %v (ok=%v), want false`, plain["strict"], ok)
	}
}

// TestStrictOmittedUnsupported: an instance with SupportsStrictTools pointing
// at false omits the "strict" key entirely from every tool (unknown-field 400
// safety), and a "prefer" tool silently falls back rather than erroring.
func TestStrictOmittedUnsupported(t *testing.T) {
	unsupported := false
	req := llm.LLMRequest{
		Model: "test-model",
		Tools: []llm.ToolDef{
			{
				Name:                "preferred",
				Schema:              json.RawMessage(`{"type":"object"}`),
				ConstrainedSampling: &llm.ConstrainedSampling{Strict: llm.StrictPrefer},
			},
		},
	}

	captured := captureBody(t, Config{SupportsStrictTools: &unsupported}, req)

	preferred := toolFunction(t, captured, "preferred")
	if _, ok := preferred["strict"]; ok {
		t.Errorf(`"preferred" tool function = %+v, want no "strict" key`, preferred)
	}
}

// TestStrictRequireUnsupportedFatal: a "require" tool on an unsupported
// instance fails before any HTTP request — a non-transient LLMErrorEvent plus
// a Done.Err wrapping llm.ErrProviderFatal carrying the exact upstream
// message, and the test server sees zero requests.
func TestStrictRequireUnsupportedFatal(t *testing.T) {
	var hits int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
	}))
	defer ts.Close()

	unsupported := false
	p, err := New(Config{
		Name:                "openai-compat",
		BaseURL:             ts.URL,
		SupportsStrictTools: &unsupported,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := llm.LLMRequest{
		Model: "test-model",
		Tools: []llm.ToolDef{
			{
				Name:                "lookup",
				Schema:              json.RawMessage(`{"type":"object"}`),
				ConstrainedSampling: &llm.ConstrainedSampling{Strict: llm.StrictRequire},
			},
		},
	}

	stream := p.Stream(context.Background(), req)

	var errEvents []llm.LLMErrorEvent
	for ev := range stream.Events {
		if e, ok := ev.(llm.LLMErrorEvent); ok {
			errEvents = append(errEvents, e)
		}
	}
	result := <-stream.Done

	if len(errEvents) != 1 {
		t.Fatalf("expected exactly one LLMErrorEvent, got %d: %+v", len(errEvents), errEvents)
	}
	if errEvents[0].Transient {
		t.Errorf("LLMErrorEvent.Transient = true, want false (fatal, non-retryable)")
	}

	if result.Err == nil {
		t.Fatal("expected Done.Err, got nil")
	}
	if !errors.Is(result.Err, llm.ErrProviderFatal) {
		t.Errorf("errors.Is(Done.Err, llm.ErrProviderFatal) = false, Done.Err = %v", result.Err)
	}
	const wantMsg = `Tool "lookup" requires JSON-schema constrained sampling, but strict tools are unsupported.`
	if got := result.Err.Error(); !strings.Contains(got, wantMsg) {
		t.Errorf("Done.Err = %q, want it to contain %q", got, wantMsg)
	}

	if hits != 0 {
		t.Fatalf("expected zero HTTP requests, got %d", hits)
	}
}
