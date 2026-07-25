// Package gemini provides an LLMProvider implementation built on the official
// Google GenAI SDK (google.golang.org/genai).
package gemini

import (
	"testing"

	llm "github.com/dev-resolute/resolute-llm-go"
)

func TestToGeminiContentsUserImage(t *testing.T) {
	msgs := []llm.Message{
		{Role: "user", Content: llm.TextContent{Text: "what is in this picture?"}},
		{Role: "user", Content: llm.ImageContent{Data: []byte{0xff, 0xd8, 0xff, 0xe0}, MimeType: "image/jpeg"}},
	}
	contents, _ := toGeminiContents(msgs)
	if len(contents) != 2 {
		t.Fatalf("contents = %d, want 2 (text turn + image turn)", len(contents))
	}
	img := contents[1]
	if img.Role != "user" {
		t.Errorf("image content role = %q, want user", img.Role)
	}
	if len(img.Parts) != 1 || img.Parts[0].InlineData == nil {
		t.Fatalf("image content parts = %+v, want one InlineData part", img.Parts)
	}
	if img.Parts[0].InlineData.MIMEType != "image/jpeg" {
		t.Errorf("mime = %q, want image/jpeg", img.Parts[0].InlineData.MIMEType)
	}
	if string(img.Parts[0].InlineData.Data) != string([]byte{0xff, 0xd8, 0xff, 0xe0}) {
		t.Errorf("data mismatch")
	}
}

func TestToGeminiContentsToolResultImages(t *testing.T) {
	msgs := []llm.Message{
		{Role: "tool", Content: llm.ToolResultContent{
			CallID: "call_1", ToolName: "read", Content: "",
			Images: []llm.ImageContent{
				{Data: []byte{1, 2}, MimeType: "image/png"},
				{Data: []byte{3, 4}, MimeType: "image/jpeg"},
			},
		}},
	}
	contents, _ := toGeminiContents(msgs)
	if len(contents) != 2 {
		t.Fatalf("contents = %d, want 2 (functionResponse + image user turn)", len(contents))
	}
	fr := contents[0]
	if fr.Parts[0].FunctionResponse == nil {
		t.Fatalf("first content is not a functionResponse: %+v", fr.Parts[0])
	}
	if got := fr.Parts[0].FunctionResponse.Response["result"]; got != "(see attached image)" {
		t.Errorf("empty-text placeholder = %v, want %q", got, "(see attached image)")
	}
	img := contents[1]
	if img.Role != "user" {
		t.Errorf("image turn role = %q, want user", img.Role)
	}
	if len(img.Parts) != 3 {
		t.Fatalf("image turn parts = %d, want 3 (marker text + 2 images)", len(img.Parts))
	}
	if img.Parts[0].Text != "Tool result image:" {
		t.Errorf("marker = %q, want %q", img.Parts[0].Text, "Tool result image:")
	}
	if img.Parts[1].InlineData == nil || img.Parts[1].InlineData.MIMEType != "image/png" {
		t.Errorf("first image part wrong: %+v", img.Parts[1])
	}
}

func TestToGeminiContentsToolResultTextWithImagesKeepsText(t *testing.T) {
	msgs := []llm.Message{
		{Role: "tool", Content: llm.ToolResultContent{
			CallID: "call_1", ToolName: "read", Content: "Read image file [image/png]",
			Images: []llm.ImageContent{{Data: []byte{1}, MimeType: "image/png"}},
		}},
	}
	contents, _ := toGeminiContents(msgs)
	fr := contents[0]
	if got := fr.Parts[0].FunctionResponse.Response["result"]; got != "Read image file [image/png]" {
		t.Errorf("result = %v, want the tool text", got)
	}
}
