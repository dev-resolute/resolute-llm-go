package gemini

import (
	"testing"

	"google.golang.org/genai"
)

func TestClientConfigVertex(t *testing.T) {
	cc := Config{Vertex: true, Project: "proj", Location: "us-central1"}.clientConfig("ignored")
	if cc.Backend != genai.BackendVertexAI {
		t.Errorf("Backend = %v, want BackendVertexAI", cc.Backend)
	}
	if cc.Project != "proj" || cc.Location != "us-central1" {
		t.Errorf("Project/Location = %q/%q, want proj/us-central1", cc.Project, cc.Location)
	}
	if cc.APIKey != "" {
		t.Errorf("APIKey = %q, want empty (Vertex uses ADC, not a key)", cc.APIKey)
	}
}

func TestClientConfigAPIKey(t *testing.T) {
	cc := Config{APIKey: "static"}.clientConfig("resolved")
	if cc.Backend == genai.BackendVertexAI {
		t.Error("Backend = BackendVertexAI, want the Gemini API path")
	}
	if cc.APIKey != "resolved" {
		t.Errorf("APIKey = %q, want the resolved key", cc.APIKey)
	}
}
