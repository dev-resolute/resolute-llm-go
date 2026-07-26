package openaicompat

import (
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

func TestXAIReasoningEffortGatedPerModel(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		thinking  llm.ThinkingLevel
		wantSet   bool
		wantValue string
	}{
		{name: "grok-3-mini accepts effort", model: "grok-3-mini", thinking: llm.ThinkingHigh, wantSet: true, wantValue: "high"},
		{name: "grok-4 reasons but effort omitted", model: "grok-4", thinking: llm.ThinkingHigh, wantSet: false},
		{name: "grok-3 off omits effort", model: "grok-3", thinking: llm.ThinkingOff, wantSet: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// given an xAI provider and a request at some thinking level
			// when the wire request is built
			body := captureClassifiedBody(t, Config{}, classifyXAI, llm.LLMRequest{
				Model:    tt.model,
				Thinking: tt.thinking,
				Messages: []llm.Message{{Role: "user", Content: llm.TextContent{Text: "hi"}}},
			})

			// then reasoning_effort is sent only for models that accept it
			effort, ok := body["reasoning_effort"]
			if ok != tt.wantSet {
				t.Fatalf("reasoning_effort present = %v, want %v (value %v)", ok, tt.wantSet, effort)
			}
			if tt.wantSet && effort != tt.wantValue {
				t.Errorf("reasoning_effort = %v, want %q", effort, tt.wantValue)
			}
		})
	}
}

func TestXAIProviderNameAndCapabilities(t *testing.T) {
	p, err := XAI(TargetConfig{APIKey: "test"})
	if err != nil {
		t.Fatalf("XAI: %v", err)
	}

	if got := p.Name(); got != "xai" {
		t.Errorf("Name() = %q, want %q", got, "xai")
	}

	tests := []struct {
		model        string
		wantThinking bool
		wantVision   bool
	}{
		{model: "grok-4", wantThinking: true, wantVision: true},
		{model: "grok-3-mini", wantThinking: true, wantVision: false},
		{model: "grok-3", wantThinking: false, wantVision: false},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			caps := p.Capabilities(tt.model)
			if caps.Thinking != tt.wantThinking {
				t.Errorf("Capabilities(%q).Thinking = %v, want %v", tt.model, caps.Thinking, tt.wantThinking)
			}
			if caps.Vision != tt.wantVision {
				t.Errorf("Capabilities(%q).Vision = %v, want %v", tt.model, caps.Vision, tt.wantVision)
			}
			if !caps.ToolCalling {
				t.Errorf("Capabilities(%q).ToolCalling = false, want true", tt.model)
			}
		})
	}
}

func TestClassifyXAI(t *testing.T) {
	tests := []struct {
		name           string
		model          string
		wantThinking   bool
		wantVision     bool
		wantReasonable bool // accepts the reasoning_effort param
	}{
		{name: "grok-4 reasons but rejects effort", model: "grok-4", wantThinking: true, wantVision: true, wantReasonable: false},
		{name: "grok-4-fast reasons but rejects effort", model: "grok-4-fast", wantThinking: true, wantVision: true, wantReasonable: false},
		{name: "grok-4.3 reasons with effort", model: "grok-4.3", wantThinking: true, wantVision: true, wantReasonable: true},
		{name: "grok-3-mini reasons with effort", model: "grok-3-mini", wantThinking: true, wantVision: false, wantReasonable: true},
		{name: "grok-3 no reasoning", model: "grok-3", wantThinking: false, wantVision: false, wantReasonable: false},
		{name: "grok-2-vision vision only", model: "grok-2-vision", wantThinking: false, wantVision: true, wantReasonable: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// given an xAI model id
			// when classified catalog-free
			got := classifyXAI(tt.model)

			// then thinking, vision, and reasoning_effort support match the model
			if got.thinking != tt.wantThinking {
				t.Errorf("classifyXAI(%q).thinking = %v, want %v", tt.model, got.thinking, tt.wantThinking)
			}
			if got.vision != tt.wantVision {
				t.Errorf("classifyXAI(%q).vision = %v, want %v", tt.model, got.vision, tt.wantVision)
			}
			if got.reasoningEffort != tt.wantReasonable {
				t.Errorf("classifyXAI(%q).reasoningEffort = %v, want %v", tt.model, got.reasoningEffort, tt.wantReasonable)
			}
		})
	}
}
