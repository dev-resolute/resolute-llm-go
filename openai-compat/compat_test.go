package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

// captureBody runs one stream against a server that records the request body and
// returns the decoded JSON, so tests can assert the exact wire shape.
func captureBody(t *testing.T, cfg Config, req llm.LLMRequest) map[string]any {
	t.Helper()
	var captured map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &captured)
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		flusher.Flush()
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer ts.Close()

	cfg.BaseURL = ts.URL
	cfg.Name = "openai-compat"
	p, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	stream := p.Stream(context.Background(), req)
	for range stream.Events {
	}
	if res := <-stream.Done; res.Err != nil {
		t.Fatalf("stream: %v", res.Err)
	}
	return captured
}

func TestDeepSeekThinkingFormat(t *testing.T) {
	tests := []struct {
		name       string
		compat     Compat
		thinking   llm.ThinkingLevel
		wantType   string
		wantEffort any // nil means "must be absent"
	}{
		{
			name:     "thinking on emits enabled, no effort when unsupported",
			compat:   Compat{ThinkingFormat: ThinkingDeepSeek},
			thinking: llm.ThinkingHigh,
			wantType: "enabled",
		},
		{
			name:     "thinking off emits disabled",
			compat:   Compat{ThinkingFormat: ThinkingDeepSeek},
			thinking: llm.ThinkingOff,
			wantType: "disabled",
		},
		{
			name:       "effort included when supported and on",
			compat:     Compat{ThinkingFormat: ThinkingDeepSeek, SupportsReasoningEffort: true},
			thinking:   llm.ThinkingMedium,
			wantType:   "enabled",
			wantEffort: "medium",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// given a deepseek-compat provider and a request at some thinking level
			// when the request body is built
			body := captureBody(t, Config{Compat: tt.compat}, llm.LLMRequest{
				Model:    "deepseek-v4-pro",
				Thinking: tt.thinking,
				Messages: []llm.Message{{Role: "user", Content: llm.TextContent{Text: "hi"}}},
			})

			// then thinking is toggled via the deepseek thinking:{type} field
			thinking, ok := body["thinking"].(map[string]any)
			if !ok {
				t.Fatalf("body.thinking missing or wrong type: %v", body["thinking"])
			}
			if thinking["type"] != tt.wantType {
				t.Errorf("thinking.type = %v, want %q", thinking["type"], tt.wantType)
			}
			if tt.wantEffort == nil {
				if _, present := body["reasoning_effort"]; present {
					t.Errorf("reasoning_effort present (%v), want absent", body["reasoning_effort"])
				}
			} else if body["reasoning_effort"] != tt.wantEffort {
				t.Errorf("reasoning_effort = %v, want %v", body["reasoning_effort"], tt.wantEffort)
			}
		})
	}
}

func TestDefaultThinkingFormatUnchanged(t *testing.T) {
	// given the default (zero) Compat — the existing OpenAI reasoning_effort path
	// when a thinking-on request is built
	body := captureBody(t, Config{}, llm.LLMRequest{
		Model:    "o3-mini",
		Thinking: llm.ThinkingHigh,
		Messages: []llm.Message{{Role: "user", Content: llm.TextContent{Text: "hi"}}},
	})

	// then reasoning_effort is sent and no deepseek thinking field appears
	if body["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v, want high", body["reasoning_effort"])
	}
	if _, present := body["thinking"]; present {
		t.Errorf("thinking field present (%v), want absent for default format", body["thinking"])
	}
}

func TestRequiresReasoningContentOnAssistantMessages(t *testing.T) {
	// given a compat that requires reasoning_content on assistant messages
	body := captureBody(t, Config{Compat: Compat{RequiresReasoningContentOnAssistantMessages: true}}, llm.LLMRequest{
		Model: "deepseek-v4-pro",
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "q"}},
			{Role: "assistant", Content: llm.TextContent{Text: "a"}},
			{Role: "assistant", Content: llm.ToolCallContent{CallID: "c1", ToolName: "t", Args: json.RawMessage(`{}`)}},
		},
	})

	// then every assistant message carries reasoning_content; the user message does not
	msgs, ok := body["messages"].([]any)
	if !ok {
		t.Fatalf("messages missing: %v", body["messages"])
	}
	for _, m := range msgs {
		mm := m.(map[string]any)
		_, has := mm["reasoning_content"]
		if mm["role"] == "assistant" && !has {
			t.Errorf("assistant message missing reasoning_content: %v", mm)
		}
		if mm["role"] == "user" && has {
			t.Errorf("user message should not have reasoning_content: %v", mm)
		}
	}
}

func TestMaxTokensSentWhenConfigured(t *testing.T) {
	// given a compat with a max-tokens cap (opencode-go pins max_tokens)
	body := captureBody(t, Config{Compat: Compat{MaxTokens: 32000}}, llm.LLMRequest{
		Model:    "deepseek-v4-pro",
		Messages: []llm.Message{{Role: "user", Content: llm.TextContent{Text: "hi"}}},
	})

	// then max_tokens is sent (JSON numbers decode as float64)
	if body["max_tokens"] != float64(32000) {
		t.Errorf("max_tokens = %v, want 32000", body["max_tokens"])
	}

	// and it is absent when unset
	body2 := captureBody(t, Config{}, llm.LLMRequest{
		Model:    "deepseek-v4-pro",
		Messages: []llm.Message{{Role: "user", Content: llm.TextContent{Text: "hi"}}},
	})
	if _, present := body2["max_tokens"]; present {
		t.Errorf("max_tokens present (%v), want absent when unconfigured", body2["max_tokens"])
	}
}

func TestChatTemplateThinkingFormat(t *testing.T) {
	tests := []struct {
		name        string
		thinking    llm.ThinkingLevel
		wantEnabled bool
	}{
		{name: "thinking on enables chat-template thinking", thinking: llm.ThinkingHigh, wantEnabled: true},
		{name: "thinking off disables chat-template thinking", thinking: llm.ThinkingOff, wantEnabled: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// given a chat-template compat provider (Qwen3/DeepSeek-R1 behind vLLM)
			// when the request body is built
			body := captureBody(t, Config{Compat: Compat{ThinkingFormat: ThinkingChatTemplate}}, llm.LLMRequest{
				Model:    "qwen3-32b",
				Thinking: tt.thinking,
				Messages: []llm.Message{{Role: "user", Content: llm.TextContent{Text: "hi"}}},
			})

			// then enable_thinking is emitted under chat_template_kwargs
			kwargs, ok := body["chat_template_kwargs"].(map[string]any)
			if !ok {
				t.Fatalf("chat_template_kwargs missing or wrong type: %v", body["chat_template_kwargs"])
			}
			if kwargs["enable_thinking"] != tt.wantEnabled {
				t.Errorf("enable_thinking = %v, want %v", kwargs["enable_thinking"], tt.wantEnabled)
			}

			// and neither reasoning_effort nor the deepseek thinking field appears
			if _, present := body["reasoning_effort"]; present {
				t.Errorf("reasoning_effort present (%v), want absent for chat-template", body["reasoning_effort"])
			}
			if _, present := body["thinking"]; present {
				t.Errorf("thinking field present (%v), want absent for chat-template", body["thinking"])
			}
		})
	}
}

func TestCapabilitiesDeepSeekReportsThinking(t *testing.T) {
	// given a provider configured for the deepseek thinking format
	p, err := New(Config{Name: "openai-compat", BaseURL: "http://x", Compat: Compat{ThinkingFormat: ThinkingDeepSeek}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// when capabilities are reported for a non-"o" model
	caps := p.Capabilities("deepseek-v4-pro")

	// then Thinking is true (the configured format implies the model reasons)
	if !caps.Thinking {
		t.Error("Thinking = false, want true for a deepseek-format provider")
	}
}
