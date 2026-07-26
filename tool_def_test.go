package llm

import (
	"strings"
	"testing"
)

func TestResolveStrictSampling(t *testing.T) {
	tests := []struct {
		name          string
		tool          ToolDef
		supported     bool
		wantStrict    bool
		wantErrSubstr string
	}{
		{
			name:       "nil config returns false and no error",
			tool:       ToolDef{Name: "test"},
			supported:  false,
			wantStrict: false,
		},
		{
			name: "prefer and supported returns true",
			tool: ToolDef{
				Name: "test",
				ConstrainedSampling: &ConstrainedSampling{
					Strict: StrictPrefer,
				},
			},
			supported:  true,
			wantStrict: true,
		},
		{
			name: "prefer and unsupported returns false",
			tool: ToolDef{
				Name: "test",
				ConstrainedSampling: &ConstrainedSampling{
					Strict: StrictPrefer,
				},
			},
			supported:  false,
			wantStrict: false,
		},
		{
			name: "require and supported returns true",
			tool: ToolDef{
				Name: "test",
				ConstrainedSampling: &ConstrainedSampling{
					Strict: StrictRequire,
				},
			},
			supported:  true,
			wantStrict: true,
		},
		{
			name: "require and unsupported returns error",
			tool: ToolDef{
				Name: "lookup",
				ConstrainedSampling: &ConstrainedSampling{
					Strict: StrictRequire,
				},
			},
			supported:     false,
			wantErrSubstr: `Tool "lookup" requires JSON-schema constrained sampling, but strict tools are unsupported.`,
		},
		{
			name: "empty strict value returns error",
			tool: ToolDef{
				Name: "test",
				ConstrainedSampling: &ConstrainedSampling{
					Strict: "",
				},
			},
			supported:     false,
			wantErrSubstr: "invalid ConstrainedSampling.Strict",
		},
		{
			name: "invalid strict value returns error",
			tool: ToolDef{
				Name: "test",
				ConstrainedSampling: &ConstrainedSampling{
					Strict: "banana",
				},
			},
			supported:     false,
			wantErrSubstr: "invalid ConstrainedSampling.Strict",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ResolveStrictSampling(tt.tool, tt.supported)

			if tt.wantErrSubstr != "" {
				if err == nil {
					t.Errorf("ResolveStrictSampling() expected error containing %q, got nil", tt.wantErrSubstr)
				} else if !strings.Contains(err.Error(), tt.wantErrSubstr) && err.Error() != tt.wantErrSubstr {
					t.Errorf("ResolveStrictSampling() error = %q, want to contain or exactly match %q", err.Error(), tt.wantErrSubstr)
				}
			} else {
				if err != nil {
					t.Errorf("ResolveStrictSampling() unexpected error: %v", err)
				}
				if got != tt.wantStrict {
					t.Errorf("ResolveStrictSampling() = %v, want %v", got, tt.wantStrict)
				}
			}
		})
	}
}
