// Package openaicompat provides an LLMProvider implementation that targets
// any OpenAI-compatible HTTP endpoint, including OpenAI, Fireworks, Ollama,
// vLLM, llama.cpp server, and LM Studio.
package openaicompat

import (
	"testing"

	llm "github.com/dev-resolute/resolute-llm-go"
)

func TestToOpenAIMessagesUserImage(t *testing.T) {
	msgs := []llm.Message{
		{Role: "user", Content: llm.ImageContent{Data: []byte("fake"), MimeType: "image/png"}},
	}
	out := toOpenAIMessages(msgs, Compat{})
	if len(out) != 1 {
		t.Fatalf("messages = %d, want 1", len(out))
	}
	m := out[0]
	if m["role"] != "user" {
		t.Errorf("role = %v, want user", m["role"])
	}
	parts, ok := m["content"].([]map[string]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("content = %#v, want one-part array", m["content"])
	}
	if parts[0]["type"] != "image_url" {
		t.Errorf("part type = %v, want image_url", parts[0]["type"])
	}
	urlObj := parts[0]["image_url"].(map[string]any)
	// base64("fake") = "ZmFrZQ==" — matches the upstream fixture value.
	if urlObj["url"] != "data:image/png;base64,ZmFrZQ==" {
		t.Errorf("url = %v, want data:image/png;base64,ZmFrZQ==", urlObj["url"])
	}
}
