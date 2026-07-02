package gemini

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
	"google.golang.org/genai"
)

func findFunctionCallPart(t *testing.T, contents []*genai.Content) *genai.Part {
	t.Helper()
	for _, c := range contents {
		for _, p := range c.Parts {
			if p.FunctionCall != nil {
				return p
			}
		}
	}
	t.Fatal("no function-call part in converted contents")
	return nil
}

func TestToGeminiContentsReplaysThoughtSignature(t *testing.T) {
	// given an assistant tool call carrying the provider's thought signature
	sig := []byte("opaque-signature-bytes")
	messages := []llm.Message{
		{Role: "user", Content: llm.TextContent{Text: "weather in Paris?"}},
		{Role: "assistant", Content: llm.ToolCallContent{
			CallID:           "call-1",
			ToolName:         "get_weather",
			Args:             json.RawMessage(`{"city":"Paris"}`),
			ThoughtSignature: sig,
		}},
	}

	// when the history is converted back to genai contents
	contents, _ := toGeminiContents(messages)

	// then the function-call part echoes the signature verbatim
	part := findFunctionCallPart(t, contents)
	if !bytes.Equal(part.ThoughtSignature, sig) {
		t.Errorf("function-call part ThoughtSignature = %q, want %q", part.ThoughtSignature, sig)
	}
}

func TestToGeminiContentsOmitsAbsentThoughtSignature(t *testing.T) {
	// given an assistant tool call without a thought signature (non-Gemini origin)
	messages := []llm.Message{
		{Role: "assistant", Content: llm.ToolCallContent{
			CallID:   "call-1",
			ToolName: "get_weather",
			Args:     json.RawMessage(`{"city":"Paris"}`),
		}},
	}

	// when the history is converted back to genai contents
	contents, _ := toGeminiContents(messages)

	// then the function-call part carries no fabricated signature
	part := findFunctionCallPart(t, contents)
	if len(part.ThoughtSignature) != 0 {
		t.Errorf("function-call part ThoughtSignature = %q, want empty", part.ThoughtSignature)
	}
}
