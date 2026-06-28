package openaicompat

import (
	"errors"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

func TestProviderNameReportsConfiguredName(t *testing.T) {
	for _, name := range []string{"xai", "mistral", "qwen", "zai"} {
		t.Run(name, func(t *testing.T) {
			p, err := New(Config{Name: name, BaseURL: "https://example.invalid/v1"})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if got := p.Name(); got != name {
				t.Errorf("Name() = %q, want %q", got, name)
			}
		})
	}
}

func TestEmptyNameRejected(t *testing.T) {
	_, err := New(Config{BaseURL: "https://api.x.ai/v1"})
	if !errors.Is(err, llm.ErrInvalidModel) {
		t.Errorf("New with empty Name: err = %v, want errors.Is ErrInvalidModel", err)
	}
}
