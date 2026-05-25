package llm

import "encoding/json"

// Message is the LLM-side unit of transcript content.
// It is provider-shaped, not user-extensible; distinct from any agent-side
// transcript Message.
type Message struct {
	Role    string
	Content Content
}

// Content is a sealed interface for the content of a Message.
type Content interface {
	isContent()
}

// TextContent carries plain text.
type TextContent struct {
	Text string
}

func (TextContent) isContent() {}

// ToolCallContent carries a tool invocation from the LLM.
type ToolCallContent struct {
	CallID   string
	ToolName string
	Args     json.RawMessage
}

func (ToolCallContent) isContent() {}

// ToolResultContent carries the result of a tool execution back to the LLM.
type ToolResultContent struct {
	CallID   string
	ToolName string
	Content  string
	Data     json.RawMessage
	IsError  bool
}

func (ToolResultContent) isContent() {}

// ThinkingContent carries reasoning/thinking content from the LLM.
type ThinkingContent struct {
	Text string
}

func (ThinkingContent) isContent() {}
