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
	// ThoughtSignature is an opaque provider token bound to this tool call
	// (Gemini 3 thought signatures). Consumers persisting the transcript must
	// carry it onto the replayed ToolCallContent; empty for providers without one.
	ThoughtSignature []byte
}

func (ToolCallStartEvent) isLLMEvent() {}

// ToolCallEndEvent signals the end of a tool call block.
type ToolCallEndEvent struct {
	CallID string
	// ToolName, Args, and ThoughtSignature carry the finalized call: for
	// providers that stream arguments incrementally (openai-compat), this
	// event — not ToolCallStartEvent — is where complete arguments appear.
	ToolName         string
	Args             json.RawMessage
	ThoughtSignature []byte
}

func (ToolCallEndEvent) isLLMEvent() {}

// StopReason describes why the assistant message ended. Mirrors upstream's
// StopReason set (types.ts:382) minus error/aborted, which this API signals
// via LLMErrorEvent / StreamResult.Err.
//
// Native finish reasons without a portable mapping (e.g. OpenAI
// content_filter) surface as StopReasonUnknown; this is not an error path —
// no LLMErrorEvent is emitted for them.
type StopReason string

const (
	StopReasonUnknown StopReason = ""
	StopReasonStop    StopReason = "stop"
	StopReasonLength  StopReason = "length"
	StopReasonToolUse StopReason = "toolUse"
)

// MessageEndEvent signals the end of the assistant's message.
// It is always the last non-error event in a successful stream.
// StopReason is StopReasonUnknown when the provider did not report one.
type MessageEndEvent struct {
	StopReason StopReason
}

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
