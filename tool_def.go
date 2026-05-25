package llm

import "encoding/json"

// ToolDef is the LLM-visible tool specification.
type ToolDef struct {
	Name        string
	Description string
	Schema      json.RawMessage
}
