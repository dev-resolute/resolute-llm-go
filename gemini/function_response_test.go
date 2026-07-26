// Package gemini provides an LLMProvider implementation built on the official
// Google GenAI SDK (google.golang.org/genai).
package gemini

import (
	"testing"

	llm "github.com/dev-resolute/resolute-llm-go"
)

func TestToGeminiContentsFunctionResponseKeys(t *testing.T) {
	msgs := []llm.Message{
		{Role: "tool", Content: llm.ToolResultContent{CallID: "c1", ToolName: "read", Content: "ok"}},
		{Role: "tool", Content: llm.ToolResultContent{CallID: "c2", ToolName: "bash", Content: "boom", IsError: true}},
	}
	contents, _ := toGeminiContents(msgs, "gemini-3.1-pro-preview")
	if len(contents) != 1 {
		t.Fatalf("contents = %d, want 1 merged user turn", len(contents))
	}
	parts := contents[0].Parts
	if len(parts) != 2 {
		t.Fatalf("parts = %d, want 2 functionResponse parts", len(parts))
	}
	if got := parts[0].FunctionResponse.Response["output"]; got != "ok" {
		t.Errorf(`success response = %v, want {"output":"ok"}`, parts[0].FunctionResponse.Response)
	}
	if got := parts[1].FunctionResponse.Response["error"]; got != "boom" {
		t.Errorf(`error response = %v, want {"error":"boom"}`, parts[1].FunctionResponse.Response)
	}
}

func TestToGeminiContentsImageRoutingByGeneration(t *testing.T) {
	// Upstream google-shared-image-tool-result-routing.test.ts, both arms.
	msgs := []llm.Message{
		{Role: "tool", Content: llm.ToolResultContent{CallID: "c1", ToolName: "a", Content: "first"}},
		{Role: "tool", Content: llm.ToolResultContent{CallID: "c2", ToolName: "b",
			Images: []llm.ImageContent{{MimeType: "image/png", Data: []byte{1}}}}},
		{Role: "tool", Content: llm.ToolResultContent{CallID: "c3", ToolName: "c", Content: "third"}},
	}

	// Gemini 3: ONE user turn, 3 functionResponse parts, image nested in part 2.
	c3, _ := toGeminiContents(msgs, "gemini-3.1-pro-preview")
	if len(c3) != 1 || len(c3[0].Parts) != 3 {
		t.Fatalf("gemini-3 = %d contents / %d parts, want 1/3", len(c3), len(c3[0].Parts))
	}
	imgFR := c3[0].Parts[1].FunctionResponse
	if len(imgFR.Parts) != 1 || imgFR.Parts[0].InlineData == nil {
		t.Errorf("gemini-3 image not nested in FunctionResponse.Parts: %+v", imgFR)
	}
	if imgFR.Response["output"] != "(see attached image)" {
		t.Errorf("image-only placeholder = %v", imgFR.Response)
	}

	// Gemini 2.5: merged FR turn (2 parts) + separate image turn + new FR turn.
	c25, _ := toGeminiContents(msgs, "gemini-2.5-flash")
	if len(c25) != 3 {
		t.Fatalf("gemini-2.5 = %d contents, want 3 (merged FR, image turn, FR)", len(c25))
	}
	if c25[1].Parts[0].Text != "Tool result image:" {
		t.Errorf("image turn lead text = %q", c25[1].Parts[0].Text)
	}
	if c25[2].Parts[0].FunctionResponse == nil {
		t.Errorf("third result did not start a new FR turn after image turn")
	}
}
