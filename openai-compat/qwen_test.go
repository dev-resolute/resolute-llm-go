package openaicompat

import (
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

func TestClassifyQwen(t *testing.T) {
	tests := []struct {
		name         string
		model        string
		wantThinking bool
		wantVision   bool
	}{
		{name: "qwen-plus hybrid thinks", model: "qwen-plus", wantThinking: true, wantVision: false},
		{name: "qwen3-max thinks", model: "qwen3-max", wantThinking: true, wantVision: false},
		{name: "qwen3-235b open thinks", model: "qwen3-235b-a22b", wantThinking: true, wantVision: false},
		{name: "qwq always reasons", model: "qwq-32b", wantThinking: true, wantVision: false},
		{name: "qwen3-coder no thinking", model: "qwen3-coder-plus", wantThinking: false, wantVision: false},
		{name: "qwen-vl vision only", model: "qwen-vl-max", wantThinking: false, wantVision: true},
		{name: "qwen2.5-vl vision only", model: "qwen2.5-vl-7b-instruct", wantThinking: false, wantVision: true},
		{name: "qwen3-vl thinks and sees", model: "qwen3-vl-plus", wantThinking: true, wantVision: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classifyQwen(tt.model)
			if got.thinking != tt.wantThinking {
				t.Errorf("classifyQwen(%q).thinking = %v, want %v", tt.model, got.thinking, tt.wantThinking)
			}
			if got.vision != tt.wantVision {
				t.Errorf("classifyQwen(%q).vision = %v, want %v", tt.model, got.vision, tt.wantVision)
			}
		})
	}
}

func TestQwenProviderName(t *testing.T) {
	p, err := Qwen(TargetConfig{APIKey: "test"})
	if err != nil {
		t.Fatalf("Qwen: %v", err)
	}
	if got := p.Name(); got != "qwen" {
		t.Errorf("Name() = %q, want %q", got, "qwen")
	}
	if caps := p.Capabilities("qwen3-vl-plus"); !caps.Thinking || !caps.Vision {
		t.Errorf("Capabilities(qwen3-vl-plus) = %+v, want Thinking+Vision", caps)
	}
}

func TestQwenThinkingFormat(t *testing.T) {
	tests := []struct {
		name     string
		thinking llm.ThinkingLevel
		want     bool
	}{
		{name: "thinking on enables top-level enable_thinking", thinking: llm.ThinkingHigh, want: true},
		{name: "thinking off disables it", thinking: llm.ThinkingOff, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// given a Qwen-dialect provider (DashScope compatible-mode)
			// when the wire request is built
			body := captureBody(t, Config{Compat: Compat{ThinkingFormat: ThinkingQwen}}, llm.LLMRequest{
				Model:    "qwen-plus",
				Thinking: tt.thinking,
				Messages: []llm.Message{{Role: "user", Content: llm.TextContent{Text: "hi"}}},
			})

			// then enable_thinking is a top-level bool
			got, ok := body["enable_thinking"]
			if !ok {
				t.Fatalf("enable_thinking absent, want %v", tt.want)
			}
			if got != tt.want {
				t.Errorf("enable_thinking = %v, want %v", got, tt.want)
			}

			// and no other thinking dialect leaks onto the wire
			if _, present := body["reasoning_effort"]; present {
				t.Errorf("reasoning_effort leaked for Qwen dialect")
			}
			if _, present := body["thinking"]; present {
				t.Errorf("thinking object leaked for Qwen dialect")
			}
			if _, present := body["chat_template_kwargs"]; present {
				t.Errorf("chat_template_kwargs leaked for Qwen dialect (DashScope wants top-level)")
			}
		})
	}
}
