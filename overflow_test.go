package llm

import (
	"errors"
	"testing"
)

func TestAsContextOverflow(t *testing.T) {
	tests := []struct {
		name      string
		err       error
		wantMatch bool
	}{
		{
			name:      "of N tokens form",
			err:       errors.New("maximum context length of 128000 tokens exceeded"),
			wantMatch: true,
		},
		{
			name:      "parenthesized N form (broadened in 0.79.2)",
			err:       errors.New("this request exceeds the maximum context length (128000)"),
			wantMatch: true,
		},
		{
			name:      "openai is-N-tokens form",
			err:       errors.New("This model's maximum context length is 8192 tokens, however you requested 9000"),
			wantMatch: true,
		},
		{
			name:      "unrelated error passes through",
			err:       errors.New("rate limit exceeded"),
			wantMatch: false,
		},
		{
			name:      "nil error",
			err:       nil,
			wantMatch: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// given a provider error
			// when classified
			got := AsContextOverflow(tt.err)

			// then errors.Is reports overflow only for context-length messages
			if matched := errors.Is(got, ErrContextOverflow); matched != tt.wantMatch {
				t.Errorf("errors.Is(AsContextOverflow(%v)) = %v, want %v", tt.err, matched, tt.wantMatch)
			}

			// and non-overflow errors are returned unchanged (nil stays nil)
			if !tt.wantMatch && got != tt.err {
				t.Errorf("non-overflow error altered: got %v, want %v", got, tt.err)
			}
		})
	}
}
