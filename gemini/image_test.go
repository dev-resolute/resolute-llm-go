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
