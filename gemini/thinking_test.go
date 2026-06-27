package gemini

import (
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

func thinkingBudget(t *testing.T, thinking llm.ThinkingLevel) (int32, bool) {
	t.Helper()
	return budgetFor(t, llm.LLMRequest{Model: "gemini-2.5-flash", Thinking: thinking})
}

func budgetFor(t *testing.T, req llm.LLMRequest) (int32, bool) {
	t.Helper()
	config := toGeminiConfig(req, nil)
	if config.ThinkingConfig == nil || config.ThinkingConfig.ThinkingBudget == nil {
		return 0, false
	}
	return *config.ThinkingConfig.ThinkingBudget, true
}

func TestThinkingMinimalMapsToSmallestBudget(t *testing.T) {
	// given a request at the minimal thinking level
	// when the provider builds the genai config
	budget, ok := thinkingBudget(t, llm.ThinkingMinimal)

	// then the thinking budget is the smallest valid positive value, distinct from Low
	if !ok {
		t.Fatalf("ThinkingConfig absent, want budget 512")
	}
	if budget != 512 {
		t.Errorf("thinking budget = %d, want 512", budget)
	}
}

func TestThinkingBudgetsOverridesActiveLevel(t *testing.T) {
	// given a per-level budget override for the active level
	req := llm.LLMRequest{
		Model:           "gemini-2.5-flash",
		Thinking:        llm.ThinkingHigh,
		ThinkingBudgets: map[llm.ThinkingLevel]int{llm.ThinkingHigh: 8000},
	}

	// when the provider builds the genai config
	budget, ok := budgetFor(t, req)

	// then the override is used instead of the High default (16000)
	if !ok {
		t.Fatalf("ThinkingConfig absent, want budget 8000")
	}
	if budget != 8000 {
		t.Errorf("thinking budget = %d, want 8000", budget)
	}
}

func TestThinkingBudgetsForOtherLevelIgnored(t *testing.T) {
	// given a per-level budget override for a level that is NOT active
	req := llm.LLMRequest{
		Model:           "gemini-2.5-flash",
		Thinking:        llm.ThinkingHigh,
		ThinkingBudgets: map[llm.ThinkingLevel]int{llm.ThinkingLow: 999},
	}

	// when the provider builds the genai config
	budget, ok := budgetFor(t, req)

	// then the active level's default (16000) is used, not the Low override
	if !ok {
		t.Fatalf("ThinkingConfig absent, want budget 16000")
	}
	if budget != 16000 {
		t.Errorf("thinking budget = %d, want 16000 (Low override must not affect High)", budget)
	}
}

func TestProviderHintsBudgetWinsOverThinkingBudgets(t *testing.T) {
	// given both a portable per-level budget and a provider-specific hint
	req := llm.LLMRequest{
		Model:           "gemini-2.5-flash",
		Thinking:        llm.ThinkingHigh,
		ThinkingBudgets: map[llm.ThinkingLevel]int{llm.ThinkingHigh: 8000},
		ProviderHints:   llm.ProviderHints{Gemini: &llm.GeminiHints{ThinkingBudget: 12000}},
	}

	// when the provider builds the genai config
	budget, ok := budgetFor(t, req)

	// then the provider-specific hint wins (12000), per the documented precedence
	if !ok {
		t.Fatalf("ThinkingConfig absent, want budget 12000")
	}
	if budget != 12000 {
		t.Errorf("thinking budget = %d, want 12000 (ProviderHints must win over ThinkingBudgets)", budget)
	}
}

func TestThinkingLevelBudgetMapping(t *testing.T) {
	tests := []struct {
		name     string
		thinking llm.ThinkingLevel
		want     int32
		wantSet  bool
	}{
		{name: "off omits ThinkingConfig", thinking: llm.ThinkingOff, wantSet: false},
		{name: "minimal", thinking: llm.ThinkingMinimal, want: 512, wantSet: true},
		{name: "low", thinking: llm.ThinkingLow, want: 1000, wantSet: true},
		{name: "medium", thinking: llm.ThinkingMedium, want: 4000, wantSet: true},
		{name: "high", thinking: llm.ThinkingHigh, want: 16000, wantSet: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			budget, ok := thinkingBudget(t, tt.thinking)
			if ok != tt.wantSet {
				t.Fatalf("ThinkingConfig present = %v, want %v", ok, tt.wantSet)
			}
			if tt.wantSet && budget != tt.want {
				t.Errorf("thinking budget = %d, want %d", budget, tt.want)
			}
		})
	}
}
