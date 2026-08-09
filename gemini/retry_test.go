package gemini

import (
	"errors"
	"fmt"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
	"google.golang.org/genai"
)

// The open-path classification for the retry ladder (upstream
// isRetryableProviderError): quota/conflict/timeout/5xx and status-less
// transport failures are transient; deterministic 4xx stay fatal.
func TestClassifyOpenError(t *testing.T) {
	for _, tt := range []struct {
		name      string
		err       error
		transient bool
		fatal     bool
	}{
		{"429 quota", genai.APIError{Code: 429, Message: "quota"}, true, false},
		{"503 unavailable", genai.APIError{Code: 503, Message: "down"}, true, false},
		{"500 internal", genai.APIError{Code: 500, Message: "boom"}, true, false},
		{"408 timeout", genai.APIError{Code: 408, Message: "slow"}, true, false},
		{"409 conflict", genai.APIError{Code: 409, Message: "conflict"}, true, false},
		{"transport error (no status)", fmt.Errorf("dial tcp: %w", errors.New("connection refused")), true, false},
		{"DNS failure (no status)", errors.New("dial tcp: lookup api.example: no such host"), true, false},
		{"400 invalid argument", genai.APIError{Code: 400, Status: "INVALID_ARGUMENT", Message: "bad"}, false, true},
		{"401 unauthorized", genai.APIError{Code: 401, Message: "unauthorized"}, false, true},
		{"403 forbidden", genai.APIError{Code: 403, Message: "forbidden"}, false, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := classifyOpenError(tt.err)
			var terr *llm.TransientError
			if isTransient := errors.As(got, &terr); isTransient != tt.transient {
				t.Errorf("classifyOpenError(%v): TransientError = %v, want %v", tt.err, isTransient, tt.transient)
			}
			if isFatal := errors.Is(got, llm.ErrProviderFatal); isFatal != tt.fatal {
				t.Errorf("classifyOpenError(%v): ErrProviderFatal = %v, want %v", tt.err, isFatal, tt.fatal)
			}
		})
	}
}
