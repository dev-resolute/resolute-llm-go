// Package openaicompat provides an LLMProvider implementation that targets
// any OpenAI-compatible HTTP endpoint, including OpenAI, Fireworks, Ollama,
// vLLM, llama.cpp server, and LM Studio.
package openaicompat

import (
	"encoding/json"
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

func TestToOpenAIMessagesToolResultImageBatching(t *testing.T) {
	img := func(b byte) []llm.ImageContent {
		return []llm.ImageContent{{Data: []byte{b}, MimeType: "image/png"}}
	}
	msgs := []llm.Message{
		{Role: "user", Content: llm.TextContent{Text: "Read the images"}},
		{Role: "assistant", Content: llm.ToolCallContent{CallID: "tool-1", ToolName: "read"}},
		{Role: "assistant", Content: llm.ToolCallContent{CallID: "tool-2", ToolName: "read"}},
		{Role: "tool", Content: llm.ToolResultContent{CallID: "tool-1", ToolName: "read",
			Content: "Read image file [image/png]", Images: img(1)}},
		{Role: "tool", Content: llm.ToolResultContent{CallID: "tool-2", ToolName: "read",
			Content: "Read image file [image/png]", Images: img(2)}},
	}
	out := toOpenAIMessages(msgs, Compat{})
	var roles []string
	for _, m := range out {
		roles = append(roles, m["role"].(string))
	}
	// Upstream pins: [user assistant tool tool user] (both calls share one
	// assistant message upstream; ours emits per-call assistant entries — the
	// invariant under test is the tail: tool, tool, then ONE user).
	n := len(out)
	if roles[n-3] != "tool" || roles[n-2] != "tool" || roles[n-1] != "user" {
		t.Fatalf("tail roles = %v, want [... tool tool user]", roles)
	}
	parts := out[n-1]["content"].([]map[string]any)
	if len(parts) != 2 {
		t.Fatalf("hoisted image parts = %d, want 2 (batched)", len(parts))
	}
	for _, p := range parts {
		if p["type"] != "image_url" {
			t.Errorf("part type = %v, want image_url", p["type"])
		}
	}
	// tool messages keep their text
	if out[n-3]["content"] != "Read image file [image/png]" {
		t.Errorf("tool text = %v, want kept", out[n-3]["content"])
	}
}

func TestToOpenAIMessagesEmptyToolResultPlaceholder(t *testing.T) {
	msgs := []llm.Message{
		{Role: "tool", Content: llm.ToolResultContent{CallID: "tool-1", ToolName: "bash", Content: ""}},
	}
	out := toOpenAIMessages(msgs, Compat{})
	if out[0]["content"] != "(no tool output)" {
		t.Errorf("content = %v, want (no tool output)", out[0]["content"])
	}
	if len(out) != 1 {
		t.Errorf("no trailing user message expected for imageless results, got %d messages", len(out))
	}
}

// TestToOpenAIMessagesEmptyToolResultWithImagePlaceholder pins the three-way
// placeholder branch: empty tool text WITH images uses "(see attached
// image)", not "(no tool output)" (upstream openai-completions.ts:1201:
// `hasText ? textResult : hasImages ? "(see attached image)" : "(no tool
// output)"`), and the image is still hoisted into a trailing user message.
func TestToOpenAIMessagesEmptyToolResultWithImagePlaceholder(t *testing.T) {
	msgs := []llm.Message{
		{Role: "tool", Content: llm.ToolResultContent{CallID: "tool-1", ToolName: "read", Content: "",
			Images: []llm.ImageContent{{Data: []byte("fake"), MimeType: "image/png"}}}},
	}
	out := toOpenAIMessages(msgs, Compat{})
	var roles []string
	for _, m := range out {
		roles = append(roles, m["role"].(string))
	}
	if len(roles) != 2 || roles[0] != "tool" || roles[1] != "user" {
		t.Fatalf("roles = %v, want [tool user]", roles)
	}
	if out[0]["content"] != "(see attached image)" {
		t.Errorf("tool content = %v, want (see attached image)", out[0]["content"])
	}
	parts, ok := out[1]["content"].([]map[string]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("hoisted content = %#v, want one-part array", out[1]["content"])
	}
	if parts[0]["type"] != "image_url" {
		t.Errorf("part type = %v, want image_url", parts[0]["type"])
	}
}

// TestStreamWireToolResultImageBatching pins the end-to-end wire shape: a
// tool-result transcript carrying an image reaches the HTTP body as a
// trailing user message whose content array holds an image_url part.
func TestStreamWireToolResultImageBatching(t *testing.T) {
	req := llm.LLMRequest{
		Model: "test-model",
		Messages: []llm.Message{
			{Role: "user", Content: llm.TextContent{Text: "Read the image"}},
			{Role: "assistant", Content: llm.ToolCallContent{CallID: "tool-1", ToolName: "read", Args: json.RawMessage(`{"path":"img.png"}`)}},
			{Role: "tool", Content: llm.ToolResultContent{CallID: "tool-1", ToolName: "read",
				Content: "Read image file [image/png]",
				Images:  []llm.ImageContent{{Data: []byte("fake"), MimeType: "image/png"}}}},
		},
	}

	captured := captureBody(t, Config{}, req)
	rawMsgs, ok := captured["messages"].([]any)
	if !ok {
		t.Fatalf("captured messages has type %T, want []any", captured["messages"])
	}

	n := len(rawMsgs)
	if n == 0 {
		t.Fatalf("captured no messages")
	}
	last, ok := rawMsgs[n-1].(map[string]any)
	if !ok {
		t.Fatalf("last message has type %T, want map[string]any", rawMsgs[n-1])
	}
	if last["role"] != "user" {
		t.Fatalf("last message role = %v, want user", last["role"])
	}
	parts, ok := last["content"].([]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("last message content = %#v, want one-part array", last["content"])
	}
	part, ok := parts[0].(map[string]any)
	if !ok || part["type"] != "image_url" {
		t.Fatalf("hoisted part = %#v, want type image_url", parts[0])
	}
	urlObj, ok := part["image_url"].(map[string]any)
	if !ok {
		t.Fatalf("image_url = %#v, want map", part["image_url"])
	}
	// base64("fake") = "ZmFrZQ==" — matches the upstream fixture value.
	if urlObj["url"] != "data:image/png;base64,ZmFrZQ==" {
		t.Errorf("url = %v, want data:image/png;base64,ZmFrZQ==", urlObj["url"])
	}
}
