package openaicompat

import (
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

func TestClassifyMistral(t *testing.T) {
	tests := []struct {
		name           string
		model          string
		wantThinking   bool
		wantVision     bool
		wantReasonable bool // accepts the reasoning_effort param
	}{
		{name: "mistral-large no reasoning no vision", model: "mistral-large-latest", wantThinking: false, wantVision: false, wantReasonable: false},
		{name: "mistral-medium hybrid reasons with effort", model: "mistral-medium-latest", wantThinking: true, wantVision: true, wantReasonable: true},
		{name: "mistral-small hybrid reasons with effort", model: "mistral-small-latest", wantThinking: true, wantVision: true, wantReasonable: true},
		{name: "magistral-medium native reasons, no effort, vision", model: "magistral-medium-latest", wantThinking: true, wantVision: true, wantReasonable: false},
		{name: "magistral-small native reasons, no effort, no vision", model: "magistral-small-latest", wantThinking: true, wantVision: false, wantReasonable: false},
		{name: "pixtral vision only", model: "pixtral-large-latest", wantThinking: false, wantVision: true, wantReasonable: false},
		{name: "codestral plain", model: "codestral-latest", wantThinking: false, wantVision: false, wantReasonable: false},
		{name: "ministral plain", model: "ministral-8b-latest", wantThinking: false, wantVision: false, wantReasonable: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classifyMistral(tt.model)
			if got.thinking != tt.wantThinking {
				t.Errorf("classifyMistral(%q).thinking = %v, want %v", tt.model, got.thinking, tt.wantThinking)
			}
			if got.vision != tt.wantVision {
				t.Errorf("classifyMistral(%q).vision = %v, want %v", tt.model, got.vision, tt.wantVision)
			}
			if got.reasoningEffort != tt.wantReasonable {
				t.Errorf("classifyMistral(%q).reasoningEffort = %v, want %v", tt.model, got.reasoningEffort, tt.wantReasonable)
			}
		})
	}
}

func TestMistralReasoningEffortGatedPerModel(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		thinking llm.ThinkingLevel
		wantSet  bool
	}{
		{name: "mistral-small accepts effort", model: "mistral-small-latest", thinking: llm.ThinkingHigh, wantSet: true},
		{name: "magistral reasons but effort omitted", model: "magistral-medium-latest", thinking: llm.ThinkingHigh, wantSet: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := captureClassifiedBody(t, Compat{}, classifyMistral, llm.LLMRequest{
				Model:    tt.model,
				Thinking: tt.thinking,
				Messages: []llm.Message{{Role: "user", Content: llm.TextContent{Text: "hi"}}},
			})
			if _, ok := body["reasoning_effort"]; ok != tt.wantSet {
				t.Errorf("reasoning_effort present = %v, want %v", ok, tt.wantSet)
			}
		})
	}
}

func TestMistralProviderName(t *testing.T) {
	p, err := Mistral(TargetConfig{APIKey: "test"})
	if err != nil {
		t.Fatalf("Mistral: %v", err)
	}
	if got := p.Name(); got != "mistral" {
		t.Errorf("Name() = %q, want %q", got, "mistral")
	}
	if caps := p.Capabilities("magistral-medium-latest"); !caps.Thinking || !caps.Vision {
		t.Errorf("Capabilities(magistral-medium) = %+v, want Thinking+Vision", caps)
	}
}
