package openaicompat

import (
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

func TestClassifyZAI(t *testing.T) {
	tests := []struct {
		name         string
		model        string
		wantThinking bool
		wantVision   bool
	}{
		{name: "glm-4.6 hybrid thinks", model: "glm-4.6", wantThinking: true, wantVision: false},
		{name: "glm-4.5 thinks", model: "glm-4.5", wantThinking: true, wantVision: false},
		{name: "glm-4.5-air thinks", model: "glm-4.5-air", wantThinking: true, wantVision: false},
		{name: "glm-4.5v thinks and sees", model: "glm-4.5v", wantThinking: true, wantVision: true},
		{name: "glm-4v-plus vision only", model: "glm-4v-plus", wantThinking: false, wantVision: true},
		{name: "glm-4-32b legacy plain", model: "glm-4-32b", wantThinking: false, wantVision: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classifyZAI(tt.model)
			if got.thinking != tt.wantThinking {
				t.Errorf("classifyZAI(%q).thinking = %v, want %v", tt.model, got.thinking, tt.wantThinking)
			}
			if got.vision != tt.wantVision {
				t.Errorf("classifyZAI(%q).vision = %v, want %v", tt.model, got.vision, tt.wantVision)
			}
		})
	}
}

func TestZAIProviderName(t *testing.T) {
	p, err := ZAI(TargetConfig{APIKey: "test"})
	if err != nil {
		t.Fatalf("ZAI: %v", err)
	}
	if got := p.Name(); got != "zai" {
		t.Errorf("Name() = %q, want %q", got, "zai")
	}
	if caps := p.Capabilities("glm-4.5v"); !caps.Thinking || !caps.Vision {
		t.Errorf("Capabilities(glm-4.5v) = %+v, want Thinking+Vision", caps)
	}
}

func TestZAIThinkingDialect(t *testing.T) {
	tests := []struct {
		name     string
		thinking llm.ThinkingLevel
		wantType string
	}{
		{name: "thinking on enables the GLM thinking object", thinking: llm.ThinkingHigh, wantType: "enabled"},
		{name: "thinking off disables it", thinking: llm.ThinkingOff, wantType: "disabled"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// given a z.ai/GLM provider (shares DeepSeek's thinking:{type} wire)
			body := captureClassifiedBody(t, Compat{ThinkingFormat: ThinkingDeepSeek}, classifyZAI, llm.LLMRequest{
				Model:    "glm-4.6",
				Thinking: tt.thinking,
				Messages: []llm.Message{{Role: "user", Content: llm.TextContent{Text: "hi"}}},
			})

			// then thinking is toggled via thinking:{type}, with no reasoning_effort
			thinking, ok := body["thinking"].(map[string]any)
			if !ok {
				t.Fatalf("thinking object missing: %v", body["thinking"])
			}
			if thinking["type"] != tt.wantType {
				t.Errorf("thinking.type = %v, want %q", thinking["type"], tt.wantType)
			}
			if _, present := body["reasoning_effort"]; present {
				t.Errorf("reasoning_effort leaked for GLM dialect")
			}
		})
	}
}
