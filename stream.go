package llm

import (
	"encoding/json"
	"time"
)

// NewEventStream creates an EventStream from the provided channels.
// Providers use this to package their internal goroutine outputs.
func NewEventStream(events <-chan LLMEvent, done <-chan StreamResult) EventStream {
	return EventStream{Events: events, Done: done}
}

// EventStream is the shared return shape for any streaming LLM call.
// Events delivers a stream of typed events and closes when the run finishes.
// Done delivers exactly one terminal StreamResult.
type EventStream struct {
	Events <-chan LLMEvent
	Done   <-chan StreamResult
}

// StreamResult is the single terminal value delivered on EventStream.Done.
type StreamResult struct {
	Messages []Message
	Err      error
}

// LLMEvent is a sealed interface for every event that flows on EventStream.Events.
// Concrete variants are defined below; discrimination is via type switch.
type LLMEvent interface {
	isLLMEvent()
}

// TextDeltaEvent carries a fragment of text output from the LLM.
type TextDeltaEvent struct {
	Delta string
}

func (TextDeltaEvent) isLLMEvent() {}

// ThinkingDeltaEvent carries a fragment of thinking/reasoning content.
type ThinkingDeltaEvent struct {
	Delta string
}

func (ThinkingDeltaEvent) isLLMEvent() {}

// ToolCallStartEvent signals the beginning of a tool call in the stream.
type ToolCallStartEvent struct {
	CallID   string
	ToolName string
	Args     json.RawMessage
}

func (ToolCallStartEvent) isLLMEvent() {}

// ToolCallEndEvent signals the end of a tool call block.
type ToolCallEndEvent struct {
	CallID string
}

func (ToolCallEndEvent) isLLMEvent() {}

// MessageEndEvent signals the end of the assistant's message.
// It is always the last non-error event in a successful stream.
type MessageEndEvent struct{}

func (MessageEndEvent) isLLMEvent() {}

// LLMErrorEvent signals an error from the provider.
type LLMErrorEvent struct {
	Error     error
	Transient bool
}

func (LLMErrorEvent) isLLMEvent() {}

// LLMRetryEvent signals that a retry attempt is being made.
type LLMRetryEvent struct {
	Provider   string
	Model      string
	Attempt    int
	NextDelay  time.Duration
	Reason     string
	ServerHint bool
}

func (LLMRetryEvent) isLLMEvent() {}

// UsageEvent carries token-usage metadata from the provider.
type UsageEvent struct {
	InputTokens  int
	OutputTokens int
}

func (UsageEvent) isLLMEvent() {}
