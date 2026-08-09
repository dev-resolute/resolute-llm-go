package gemini

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
	"google.golang.org/genai"
)

// findPart locates the first part matching pred across converted contents.
func findPart(t *testing.T, contents []*genai.Content, pred func(*genai.Part) bool) *genai.Part {
	t.Helper()
	for _, c := range contents {
		for _, p := range c.Parts {
			if pred(p) {
				return p
			}
		}
	}
	t.Fatal("no matching part in converted contents")
	return nil
}

func countParts(contents []*genai.Content) int {
	n := 0
	for _, c := range contents {
		n += len(c.Parts)
	}
	return n
}

// The text-delta rule: text parts emit (even empty ones carrying only a
// signature); function-call parts never do — their signature rides the
// ToolCall events (upstream keys the text branch on part.text presence).
func TestEmitsTextDelta(t *testing.T) {
	sig := []byte("sig")
	for _, tt := range []struct {
		name string
		part *genai.Part
		want bool
	}{
		{"plain text", &genai.Part{Text: "hi"}, true},
		{"signed empty text", &genai.Part{Text: "", ThoughtSignature: sig}, true},
		{"unsigned empty text", &genai.Part{Text: ""}, false},
		{"signed function call", &genai.Part{FunctionCall: &genai.FunctionCall{Name: "f"}, ThoughtSignature: sig}, false},
		{"unsigned function call", &genai.Part{FunctionCall: &genai.FunctionCall{Name: "f"}}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := emitsTextDelta(tt.part); got != tt.want {
				t.Errorf("emitsTextDelta(%+v) = %v, want %v", tt.part, got, tt.want)
			}
		})
	}
}

// --- upstream #7494: tool-call IDs must survive history conversion on Gemini 3 ---

func TestToGeminiContentsReplaysToolCallIDOnGemini3(t *testing.T) {
	// given a full tool-call round trip in history
	messages := []llm.Message{
		{Role: "assistant", Content: llm.ToolCallContent{
			CallID:   "call-42",
			ToolName: "get_weather",
			Args:     json.RawMessage(`{"city":"Paris"}`),
		}},
		{Role: "tool", Content: llm.ToolResultContent{
			CallID:   "call-42",
			ToolName: "get_weather",
			Content:  "sunny",
		}},
	}

	// when converted for a Gemini 3 model
	contents, _ := toGeminiContents(messages, "gemini-3.1-pro-preview")

	// then both the call and its response carry the ID for signed replay
	call := findPart(t, contents, func(p *genai.Part) bool { return p.FunctionCall != nil })
	if call.FunctionCall.ID != "call-42" {
		t.Errorf("FunctionCall.ID = %q, want %q", call.FunctionCall.ID, "call-42")
	}
	resp := findPart(t, contents, func(p *genai.Part) bool { return p.FunctionResponse != nil })
	if resp.FunctionResponse.ID != "call-42" {
		t.Errorf("FunctionResponse.ID = %q, want %q", resp.FunctionResponse.ID, "call-42")
	}
}

func TestToGeminiContentsOmitsToolCallIDBeforeGemini3(t *testing.T) {
	// given the same round trip in history
	messages := []llm.Message{
		{Role: "assistant", Content: llm.ToolCallContent{
			CallID:   "call-42",
			ToolName: "get_weather",
			Args:     json.RawMessage(`{"city":"Paris"}`),
		}},
		{Role: "tool", Content: llm.ToolResultContent{
			CallID:   "call-42",
			ToolName: "get_weather",
			Content:  "sunny",
		}},
	}

	for _, model := range []string{"gemini-2.5-flash", "gemini-1.5-pro", "gemma-4-27b"} {
		// when converted for a pre-Gemini-3 model
		contents, _ := toGeminiContents(messages, model)

		// then no IDs are emitted (those backends reject or ignore them)
		call := findPart(t, contents, func(p *genai.Part) bool { return p.FunctionCall != nil })
		if call.FunctionCall.ID != "" {
			t.Errorf("model %q: FunctionCall.ID = %q, want empty", model, call.FunctionCall.ID)
		}
		resp := findPart(t, contents, func(p *genai.Part) bool { return p.FunctionResponse != nil })
		if resp.FunctionResponse.ID != "" {
			t.Errorf("model %q: FunctionResponse.ID = %q, want empty", model, resp.FunctionResponse.ID)
		}
	}
}

// --- upstream #7362: signed empty text/thinking blocks must survive conversion ---

func TestToGeminiContentsKeepsSignedEmptyAssistantText(t *testing.T) {
	// given an assistant text block whose visible text is empty but carries a
	// thought signature (Gemini attaches signatures to such parts and requires
	// them echoed back)
	sig := []byte("sig-bytes")
	messages := []llm.Message{
		{Role: "assistant", Content: llm.TextContent{Text: "", ThoughtSignature: sig}},
	}

	contents, _ := toGeminiContents(messages, "gemini-3.1-pro-preview")

	part := findPart(t, contents, func(p *genai.Part) bool { return p.FunctionCall == nil })
	if !bytes.Equal(part.ThoughtSignature, sig) {
		t.Errorf("part ThoughtSignature = %q, want %q", part.ThoughtSignature, sig)
	}
	if part.Thought {
		t.Error("text part must not be marked Thought")
	}
}

func TestToGeminiContentsDropsUnsignedEmptyAssistantText(t *testing.T) {
	// given assistant text blocks with no signature and no visible content
	messages := []llm.Message{
		{Role: "assistant", Content: llm.TextContent{Text: ""}},
		{Role: "assistant", Content: llm.TextContent{Text: "  \n"}},
		{Role: "assistant", Content: llm.TextContent{Text: "real answer"}},
	}

	contents, _ := toGeminiContents(messages, "gemini-3.1-pro-preview")

	// then only the substantive text survives
	if n := countParts(contents); n != 1 {
		t.Fatalf("converted parts = %d, want 1 (empty unsigned blocks dropped)", n)
	}
	part := findPart(t, contents, func(p *genai.Part) bool { return p.FunctionCall == nil })
	if part.Text != "real answer" {
		t.Errorf("part Text = %q, want %q", part.Text, "real answer")
	}
}

func TestToGeminiContentsKeepsUnsignedEmptyUserText(t *testing.T) {
	// given an empty user text (upstream's empty-block filter is assistant-only)
	messages := []llm.Message{
		{Role: "user", Content: llm.TextContent{Text: ""}},
	}

	contents, _ := toGeminiContents(messages, "gemini-3.1-pro-preview")

	if n := countParts(contents); n != 1 {
		t.Fatalf("converted parts = %d, want 1 (user text is never filtered)", n)
	}
}

func TestToGeminiContentsThinkingBecomesSignedThoughtPart(t *testing.T) {
	// given an assistant thinking block with a thought signature
	sig := []byte("think-sig")
	messages := []llm.Message{
		{Role: "assistant", Content: llm.ThinkingContent{Text: "let me reason", ThoughtSignature: sig}},
	}

	contents, _ := toGeminiContents(messages, "gemini-3.1-pro-preview")

	// then it replays as a thought part carrying the signature verbatim
	part := findPart(t, contents, func(p *genai.Part) bool { return p.FunctionCall == nil })
	if !part.Thought {
		t.Error("thinking block must convert to a Thought part, not flattened text")
	}
	if part.Text != "let me reason" {
		t.Errorf("part Text = %q, want %q", part.Text, "let me reason")
	}
	if !bytes.Equal(part.ThoughtSignature, sig) {
		t.Errorf("part ThoughtSignature = %q, want %q", part.ThoughtSignature, sig)
	}
}

func TestToGeminiContentsThinkingEmptyRules(t *testing.T) {
	// given empty thinking blocks, one signed and one not
	sig := []byte("think-sig")
	messages := []llm.Message{
		{Role: "assistant", Content: llm.ThinkingContent{Text: ""}},
		{Role: "assistant", Content: llm.ThinkingContent{Text: "", ThoughtSignature: sig}},
	}

	contents, _ := toGeminiContents(messages, "gemini-3.1-pro-preview")

	// then only the signed block survives
	if n := countParts(contents); n != 1 {
		t.Fatalf("converted parts = %d, want 1 (unsigned empty thinking dropped)", n)
	}
	part := findPart(t, contents, func(p *genai.Part) bool { return p.FunctionCall == nil })
	if !part.Thought || !bytes.Equal(part.ThoughtSignature, sig) {
		t.Errorf("surviving part = {Thought: %v, sig: %q}, want {Thought: true, sig: %q}",
			part.Thought, part.ThoughtSignature, sig)
	}
}
