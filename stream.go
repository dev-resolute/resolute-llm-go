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
	// ThoughtSignature is an opaque provider token bound to the text part this
	// delta belongs to (Gemini thought signatures). It typically appears on only
	// one delta of a part — possibly one with an empty Delta — so consumers
	// assembling a message retain the last non-empty value (upstream
	// retainThoughtSignature); empty for providers without one.
	ThoughtSignature []byte
}

func (TextDeltaEvent) isLLMEvent() {}

// ThinkingDeltaEvent carries a fragment of thinking/reasoning content.
type ThinkingDeltaEvent struct {
	Delta string
	// ThoughtSignature behaves as on TextDeltaEvent, for thinking parts.
	ThoughtSignature []byte
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
// StopReason set (types.ts) minus error/aborted/pending/deferred: terminal
// provider stops are fatal errors, not stop reasons (see below), upstream's
// pending is an interim partial-message state this event shape does not
// expose, and deferred responses are out of scope.
//
// Since 0.13.0 (upstream #7272 parity): native finish reasons without a
// portable mapping (Gemini SAFETY/RECITATION/..., OpenAI
// content_filter/network_error, or a genuinely unknown reason) surface as a
// fatal LLMErrorEvent wrapping ErrProviderStop — never as a successful
// StopReasonUnknown. A stream that ends without any finish reason is a
// protocol error wrapping ErrMalformedResponse, unless the provider is known
// to omit it (openai-compat Compat.SupportsFinishReason), in which case the
// reason is inferred from content.
type StopReason string

const (
	// StopReasonUnknown is the zero value. Providers no longer emit it on a
	// successful MessageEndEvent; it remains for unmapped custom-provider use.
	StopReasonUnknown StopReason = ""
	StopReasonStop    StopReason = "stop"
	StopReasonLength  StopReason = "length"
	StopReasonToolUse StopReason = "toolUse"
)

// MessageEndEvent signals the end of the assistant's message.
// It is always the last non-error event in a successful stream; a stream that
// terminates with a provider stop (or missing finish reason) emits a fatal
// LLMErrorEvent instead.
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
