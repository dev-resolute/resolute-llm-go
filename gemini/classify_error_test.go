package gemini

import (
	"errors"
	"fmt"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
	"google.golang.org/genai"
)

// TestClassifyStreamError pins LLM-11: deterministic client errors are wrapped
// in llm.ErrProviderFatal so retry ladders stop retrying them; quota, server,
// transport, and context-overflow errors pass through unclassified.
func TestClassifyStreamError(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name      string
		err       error
		wantFatal bool
	}{
		{name: "nil", err: nil, wantFatal: false},
		{
			name:      "400 invalid argument",
			err:       genai.APIError{Code: 400, Status: "INVALID_ARGUMENT", Message: "Function call is missing a thought_signature"},
			wantFatal: true,
		},
		{
			name:      "400 failed precondition",
			err:       genai.APIError{Code: 400, Status: "FAILED_PRECONDITION", Message: "billing required"},
			wantFatal: true,
		},
		{
			name:      "401 unauthenticated",
			err:       genai.APIError{Code: 401, Status: "UNAUTHENTICATED", Message: "invalid api key"},
			wantFatal: true,
		},
		{
			name:      "403 permission denied",
			err:       genai.APIError{Code: 403, Status: "PERMISSION_DENIED", Message: "key lacks access"},
			wantFatal: true,
		},
		{
			name:      "404 not found",
			err:       genai.APIError{Code: 404, Status: "NOT_FOUND", Message: "model not found"},
			wantFatal: true,
		},
		{
			name:      "status only, no http code",
			err:       genai.APIError{Status: "INVALID_ARGUMENT", Message: "bad request"},
			wantFatal: true,
		},
		{
			name:      "wrapped api error",
			err:       fmt.Errorf("stream chunk: %w", genai.APIError{Code: 400, Status: "INVALID_ARGUMENT"}),
			wantFatal: true,
		},
		{
			name:      "429 resource exhausted stays transient",
			err:       genai.APIError{Code: 429, Status: "RESOURCE_EXHAUSTED", Message: "quota exceeded"},
			wantFatal: false,
		},
		{
			name:      "500 internal stays transient",
			err:       genai.APIError{Code: 500, Status: "INTERNAL", Message: "server error"},
			wantFatal: false,
		},
		{
			name:      "503 unavailable stays transient",
			err:       genai.APIError{Code: 503, Status: "UNAVAILABLE", Message: "overloaded"},
			wantFatal: false,
		},
		{
			name:      "plain transport error stays transient",
			err:       errors.New("connection reset by peer"),
			wantFatal: false,
		},
		{
			name:      "context overflow 400 passes through for LLM-8 handling",
			err:       genai.APIError{Code: 400, Status: "INVALID_ARGUMENT", Message: "input exceeds the maximum context length of 1048576 tokens"},
			wantFatal: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := classifyStreamError(tt.err)
			if gotFatal := errors.Is(got, llm.ErrProviderFatal); gotFatal != tt.wantFatal {
				t.Errorf("classifyStreamError(%v) fatal = %v, want %v (got error: %v)", tt.err, gotFatal, tt.wantFatal, got)
			}
			if tt.err == nil {
				if got != nil {
					t.Errorf("classifyStreamError(nil) = %v, want nil", got)
				}
				return
			}
			// The original error must stay reachable for errors.As/Is chains.
			var apiErr genai.APIError
			if errors.As(tt.err, &apiErr) {
				var gotAPIErr genai.APIError
				if !errors.As(got, &gotAPIErr) || gotAPIErr.Code != apiErr.Code {
					t.Errorf("classifyStreamError(%v) lost the underlying genai.APIError: %v", tt.err, got)
				}
			}
			// Overflow errors must still classify as overflow downstream.
			if tt.name == "context overflow 400 passes through for LLM-8 handling" {
				if !errors.Is(llm.AsContextOverflow(got), llm.ErrContextOverflow) {
					t.Errorf("overflow error no longer classifiable by AsContextOverflow: %v", got)
				}
			}
		})
	}
}
