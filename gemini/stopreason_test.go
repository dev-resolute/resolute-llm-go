package gemini

import (
	"errors"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
	"google.golang.org/genai"
)

func TestClassifyGeminiStop(t *testing.T) {
	for _, tt := range []struct {
		name        string
		reason      genai.FinishReason
		sawToolCall bool
		want        llm.StopReason
		wantErr     error
	}{
		{"stop", genai.FinishReasonStop, false, llm.StopReasonStop, nil},
		{"stop with tool calls", genai.FinishReasonStop, true, llm.StopReasonToolUse, nil},
		{"max tokens", genai.FinishReasonMaxTokens, false, llm.StopReasonLength, nil},
		// Length wins over tool use: a MAX_TOKENS-truncated message's calls
		// may be incomplete (upstream #6285).
		{"max tokens with tool calls", genai.FinishReasonMaxTokens, true, llm.StopReasonLength, nil},
		// #7272: every other named terminal reason is an error, and the error
		// wins over tool use — a SAFETY stop mid-call means arguments may be
		// incomplete, so a toolUse mapping would execute borked calls.
		{"safety", genai.FinishReasonSafety, false, "", llm.ErrProviderStop},
		{"safety with tool calls", genai.FinishReasonSafety, true, "", llm.ErrProviderStop},
		{"recitation", genai.FinishReasonRecitation, false, "", llm.ErrProviderStop},
		{"language", genai.FinishReasonLanguage, false, "", llm.ErrProviderStop},
		{"other", genai.FinishReasonOther, false, "", llm.ErrProviderStop},
		{"blocklist", genai.FinishReasonBlocklist, false, "", llm.ErrProviderStop},
		{"prohibited content", genai.FinishReasonProhibitedContent, false, "", llm.ErrProviderStop},
		{"spii", genai.FinishReasonSPII, false, "", llm.ErrProviderStop},
		{"malformed function call", genai.FinishReasonMalformedFunctionCall, false, "", llm.ErrProviderStop},
		{"image safety", genai.FinishReasonImageSafety, false, "", llm.ErrProviderStop},
		{"unexpected tool call", genai.FinishReasonUnexpectedToolCall, false, "", llm.ErrProviderStop},
		{"image prohibited content", genai.FinishReasonImageProhibitedContent, false, "", llm.ErrProviderStop},
		{"no image", genai.FinishReasonNoImage, false, "", llm.ErrProviderStop},
		{"image recitation", genai.FinishReasonImageRecitation, false, "", llm.ErrProviderStop},
		{"image other", genai.FinishReasonImageOther, false, "", llm.ErrProviderStop},
		{"unspecified", genai.FinishReasonUnspecified, false, "", llm.ErrProviderStop},
		// A genuinely unknown reason is an error too (upstream throws
		// "Unhandled stop reason").
		{"unknown string", genai.FinishReason("ALIEN_REASON"), false, "", llm.ErrProviderStop},
		// A stream that never reported a finish reason is a protocol error
		// (upstream's pending invariant), not a silent unknown stop.
		{"missing", genai.FinishReason(""), false, "", llm.ErrMalformedResponse},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := classifyGeminiStop(tt.reason, tt.sawToolCall)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Errorf("classifyGeminiStop(%q, %v) err = %v, want errors.Is %v",
						tt.reason, tt.sawToolCall, err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("classifyGeminiStop(%q, %v) unexpected err = %v", tt.reason, tt.sawToolCall, err)
			}
			if got != tt.want {
				t.Errorf("classifyGeminiStop(%q, %v) = %q, want %q", tt.reason, tt.sawToolCall, got, tt.want)
			}
		})
	}
}
