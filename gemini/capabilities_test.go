package gemini

import (
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
	"google.golang.org/genai"
)

func TestClassifyGemini(t *testing.T) {
	tests := []struct {
		name  string
		model string
		want  geminiClass
	}{
		{name: "2.5 pro", model: "gemini-2.5-pro", want: class25},
		{name: "2.5 flash", model: "gemini-2.5-flash", want: class25},
		{name: "3 pro", model: "gemini-3-pro", want: class3Pro},
		{name: "3.1 pro", model: "gemini-3.1-pro", want: class3Pro},
		{name: "3 flash", model: "gemini-3-flash", want: class3Flash},
		{name: "3.5 flash", model: "gemini-3.5-flash", want: class3Flash},
		{name: "flash-latest alias", model: "gemini-flash-latest", want: class3Flash},
		{name: "flash-lite-latest alias", model: "gemini-flash-lite-latest", want: class3Flash},
		{name: "gemma 4 dashed", model: "gemma-4", want: classGemma4},
		{name: "gemma 4 joined", model: "gemma4", want: classGemma4},
		{name: "legacy 1.5", model: "gemini-1.5-pro", want: classLegacy},
		{name: "unknown", model: "some-other-model", want: classLegacy},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// given a model string
			// when classified by generation
			got := classifyGemini(tt.model)

			// then the expected generation class is returned
			if got != tt.want {
				t.Errorf("classifyGemini(%q) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

func TestCapabilitiesByGeneration(t *testing.T) {
	p := &Provider{}
	tests := []struct {
		name         string
		model        string
		wantThinking bool
		wantVision   bool
	}{
		{name: "2.5 pro", model: "gemini-2.5-pro", wantThinking: true, wantVision: true},
		{name: "2.5 flash", model: "gemini-2.5-flash", wantThinking: true, wantVision: true},
		{name: "3.1 pro", model: "gemini-3.1-pro", wantThinking: true, wantVision: true},
		{name: "3.5 flash", model: "gemini-3.5-flash", wantThinking: true, wantVision: true},
		{name: "flash-latest", model: "gemini-flash-latest", wantThinking: true, wantVision: true},
		{name: "legacy 1.5", model: "gemini-1.5-pro", wantThinking: false, wantVision: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// given a model string
			// when capabilities are reported
			caps := p.Capabilities(tt.model)

			// then Thinking and Vision match the generation
			if caps.Thinking != tt.wantThinking {
				t.Errorf("Thinking(%q) = %v, want %v", tt.model, caps.Thinking, tt.wantThinking)
			}
			if caps.Vision != tt.wantVision {
				t.Errorf("Vision(%q) = %v, want %v", tt.model, caps.Vision, tt.wantVision)
			}
		})
	}
}

func TestThinkingConfigMechanismByGeneration(t *testing.T) {
	t.Run("gemini 2.5 uses thinkingBudget, not thinkingLevel", func(t *testing.T) {
		// given a Gemini 2.5 model with thinking on
		// when the genai config is built
		cfg, err := toGeminiConfig(llm.LLMRequest{Model: "gemini-2.5-flash", Thinking: llm.ThinkingMedium}, nil)
		if err != nil {
			t.Fatalf("toGeminiConfig: unexpected error: %v", err)
		}

		// then it carries a thinkingBudget, no thinkingLevel, and includes thoughts
		tc := cfg.ThinkingConfig
		if tc == nil {
			t.Fatal("ThinkingConfig is nil, want budget config")
		}
		if tc.ThinkingBudget == nil || *tc.ThinkingBudget != 4000 {
			t.Errorf("ThinkingBudget = %v, want 4000", tc.ThinkingBudget)
		}
		if tc.ThinkingLevel != "" {
			t.Errorf("ThinkingLevel = %q, want empty for Gemini 2.5", tc.ThinkingLevel)
		}
		if !tc.IncludeThoughts {
			t.Error("IncludeThoughts = false, want true when thinking is on")
		}
	})

	t.Run("gemini 3 uses thinkingLevel, not thinkingBudget", func(t *testing.T) {
		// given a Gemini 3 model with thinking on
		// when the genai config is built
		cfg, err := toGeminiConfig(llm.LLMRequest{Model: "gemini-3.5-flash", Thinking: llm.ThinkingHigh}, nil)
		if err != nil {
			t.Fatalf("toGeminiConfig: unexpected error: %v", err)
		}

		// then it carries a thinkingLevel enum, no budget, and includes thoughts
		tc := cfg.ThinkingConfig
		if tc == nil {
			t.Fatal("ThinkingConfig is nil, want level config")
		}
		if tc.ThinkingLevel != genai.ThinkingLevelHigh {
			t.Errorf("ThinkingLevel = %q, want %q", tc.ThinkingLevel, genai.ThinkingLevelHigh)
		}
		if tc.ThinkingBudget != nil {
			t.Errorf("ThinkingBudget = %v, want nil for Gemini 3", *tc.ThinkingBudget)
		}
		if !tc.IncludeThoughts {
			t.Error("IncludeThoughts = false, want true when thinking is on")
		}
	})

	t.Run("gemini 3 level mapping", func(t *testing.T) {
		levels := map[llm.ThinkingLevel]genai.ThinkingLevel{
			llm.ThinkingMinimal: genai.ThinkingLevelMinimal,
			llm.ThinkingLow:     genai.ThinkingLevelLow,
			llm.ThinkingMedium:  genai.ThinkingLevelMedium,
			llm.ThinkingHigh:    genai.ThinkingLevelHigh,
		}
		for portable, want := range levels {
			cfg, err := toGeminiConfig(llm.LLMRequest{Model: "gemini-3.1-pro", Thinking: portable}, nil)
			if err != nil {
				t.Fatalf("toGeminiConfig: unexpected error: %v", err)
			}
			if cfg.ThinkingConfig == nil || cfg.ThinkingConfig.ThinkingLevel != want {
				t.Errorf("portable %v → ThinkingLevel %v, want %v", portable, cfg.ThinkingConfig.ThinkingLevel, want)
			}
		}
	})
}

func TestThinkingOffByGeneration(t *testing.T) {
	t.Run("gemini 2.5 off omits ThinkingConfig", func(t *testing.T) {
		cfg, err := toGeminiConfig(llm.LLMRequest{Model: "gemini-2.5-flash", Thinking: llm.ThinkingOff}, nil)
		if err != nil {
			t.Fatalf("toGeminiConfig: unexpected error: %v", err)
		}
		if cfg.ThinkingConfig != nil {
			t.Errorf("ThinkingConfig = %+v, want nil for Gemini 2.5 off", cfg.ThinkingConfig)
		}
	})

	t.Run("gemini 3 pro off uses lowest level without thoughts", func(t *testing.T) {
		// Gemini 3 cannot fully disable thinking: lowest level, thoughts hidden.
		cfg, err := toGeminiConfig(llm.LLMRequest{Model: "gemini-3.1-pro", Thinking: llm.ThinkingOff}, nil)
		if err != nil {
			t.Fatalf("toGeminiConfig: unexpected error: %v", err)
		}
		tc := cfg.ThinkingConfig
		if tc == nil {
			t.Fatal("ThinkingConfig is nil, want disabled-level config for Gemini 3 pro")
		}
		if tc.ThinkingLevel != genai.ThinkingLevelLow {
			t.Errorf("ThinkingLevel = %q, want LOW for Gemini 3 pro off", tc.ThinkingLevel)
		}
		if tc.IncludeThoughts {
			t.Error("IncludeThoughts = true, want false when off")
		}
	})

	t.Run("gemini 3 flash off uses minimal level without thoughts", func(t *testing.T) {
		cfg, err := toGeminiConfig(llm.LLMRequest{Model: "gemini-3.5-flash", Thinking: llm.ThinkingOff}, nil)
		if err != nil {
			t.Fatalf("toGeminiConfig: unexpected error: %v", err)
		}
		tc := cfg.ThinkingConfig
		if tc == nil {
			t.Fatal("ThinkingConfig is nil, want disabled-level config for Gemini 3 flash")
		}
		if tc.ThinkingLevel != genai.ThinkingLevelMinimal {
			t.Errorf("ThinkingLevel = %q, want MINIMAL for Gemini 3 flash off", tc.ThinkingLevel)
		}
		if tc.IncludeThoughts {
			t.Error("IncludeThoughts = true, want false when off")
		}
	})
}
