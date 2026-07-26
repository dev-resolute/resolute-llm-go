// Package mock provides a first-class MockProvider for testing code that
// consumes pi-llm-go.
package mock

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/dev-resolute/resolute-llm-go"
)

// Matcher determines whether a script step applies to the current request.
type Matcher interface {
	Match(messages []llm.Message) bool
}

// exactMatcher matches when the concatenated text of all user messages equals s.
type exactMatcher struct {
	s string
}

func (m exactMatcher) Match(messages []llm.Message) bool {
	var text string
	for _, msg := range messages {
		if msg.Role == "user" {
			if tc, ok := msg.Content.(llm.TextContent); ok {
				text += tc.Text
			}
		}
	}
	return text == m.s
}

// Exact returns a Matcher that matches the exact user prompt string.
func Exact(s string) Matcher { return exactMatcher{s: s} }

// regexMatcher matches when the concatenated user text matches re.
type regexMatcher struct {
	re *regexp.Regexp
}

func (m regexMatcher) Match(messages []llm.Message) bool {
	var text string
	for _, msg := range messages {
		if msg.Role == "user" {
			if tc, ok := msg.Content.(llm.TextContent); ok {
				text += tc.Text
			}
		}
	}
	return m.re.MatchString(text)
}

// Regex returns a Matcher that matches user text against a regular expression.
func Regex(pattern string) (Matcher, error) {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, fmt.Errorf("invalid regex %q: %w", pattern, err)
	}
	return regexMatcher{re: re}, nil
}

// Predicate returns a Matcher that uses a custom function.
func Predicate(fn func(messages []llm.Message) bool) Matcher { return predicateMatcher{fn: fn} }

type predicateMatcher struct {
	fn func(messages []llm.Message) bool
}

func (m predicateMatcher) Match(messages []llm.Message) bool { return m.fn(messages) }

// LastUser returns a Matcher that matches only the text of the last user message.
func LastUser(s string) Matcher { return lastUserMatcher{s: s} }

type lastUserMatcher struct {
	s string
}

func (m lastUserMatcher) Match(messages []llm.Message) bool {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			if tc, ok := messages[i].Content.(llm.TextContent); ok {
				return strings.TrimSpace(tc.Text) == m.s
			}
		}
	}
	return false
}

// scriptStep is a single step in the mock's playback script.
type scriptStep struct {
	matcher   Matcher
	response  response
	callIndex int // for diagnostic messages
}

// response describes what the mock should emit when a step matches.
type response struct {
	events      []llm.LLMEvent
	err         error
	delay       time.Duration
	streamError error // emitted as LLMErrorEvent{Transient: false}
	resultMsgs  []llm.Message
	status      int
	respHeaders map[string]string
}

// MockProvider is a scripted LLMProvider for tests.
type MockProvider struct {
	name            string
	scripts         []scriptStep
	called          int
	mu              sync.Mutex
	recordHeaders   bool
	recordedHeaders map[string]string
}

// New creates a fresh MockProvider with the given name.
func New(name string) *MockProvider {
	if name == "" {
		name = "mock"
	}
	return &MockProvider{name: name}
}

// RecordHeaders enables header recording for the next Stream call.
func (m *MockProvider) RecordHeaders() *MockProvider {
	m.recordHeaders = true
	return m
}

// RecordedHeaders returns the headers received during the last Stream call.
func (m *MockProvider) RecordedHeaders() map[string]string {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make(map[string]string, len(m.recordedHeaders))
	for k, v := range m.recordedHeaders {
		out[k] = v
	}
	return out
}

// OnPrompt starts a fluent builder for a script step triggered by the given matcher.
func (m *MockProvider) OnPrompt(matcher Matcher) *ResponseBuilder {
	return &ResponseBuilder{mock: m, matcher: matcher}
}

// OnToolResult starts a fluent builder for a script step triggered when the
// most recent request contains a tool result for the named tool.
func (m *MockProvider) OnToolResult(toolName string, matcher Matcher) *ResponseBuilder {
	return &ResponseBuilder{mock: m, matcher: toolResultMatcher{toolName: toolName, inner: matcher}}
}

// OnAny matches any request.
func (m *MockProvider) OnAny() *ResponseBuilder {
	return &ResponseBuilder{mock: m, matcher: Predicate(func([]llm.Message) bool { return true })}
}

type toolResultMatcher struct {
	toolName string
	inner    Matcher
}

func (t toolResultMatcher) Match(messages []llm.Message) bool {
	found := false
	for _, msg := range messages {
		if tr, ok := msg.Content.(llm.ToolResultContent); ok && tr.CallID != "" {
			found = true
			break
		}
	}
	if !found {
		return false
	}
	return t.inner.Match(messages)
}

// Name implements llm.LLMProvider.
func (m *MockProvider) Name() string { return m.name }

// Capabilities implements llm.LLMProvider.
func (m *MockProvider) Capabilities(model string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{
		Streaming:         true,
		ToolCalling:       true,
		ParallelToolCalls: true,
		Thinking:          true,
		PromptCaching:     false,
		Vision:            false,
	}
}

// Stream implements llm.LLMProvider.
func (m *MockProvider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	return llm.Run(ctx, req, m.produce)
}

func (m *MockProvider) produce(ctx context.Context, req llm.LLMRequest, emit func(llm.LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]llm.Message, error) {
	m.mu.Lock()
	if m.called >= len(m.scripts) {
		m.mu.Unlock()
		return nil, fmt.Errorf("mock: unexpected call #%d (no script steps remaining): %w", m.called+1, llm.ErrProviderFatal)
	}

	step := m.scripts[m.called]
	m.called++
	m.mu.Unlock()

	if !step.matcher.Match(req.Messages) {
		return nil, fmt.Errorf("mock: call #%d did not match script step %d: %w", m.called, step.callIndex, llm.ErrProviderFatal)
	}

	if m.recordHeaders {
		m.mu.Lock()
		m.recordedHeaders = headers
		m.mu.Unlock()
	}

	if step.response.delay > 0 {
		select {
		case <-time.After(step.response.delay):
		case <-ctx.Done():
			return nil, context.Cause(ctx)
		}
	}

	for _, ev := range step.response.events {
		if err := emit(ev); err != nil {
			return nil, err
		}
	}

	if step.response.streamError != nil {
		if err := emit(llm.LLMErrorEvent{Error: step.response.streamError, Transient: false}); err != nil {
			return nil, err
		}
	}

	if setResponseMeta != nil {
		status := step.response.status
		if status == 0 {
			status = 200
		}
		respHeaders := step.response.respHeaders
		if respHeaders == nil {
			respHeaders = map[string]string{}
		}
		setResponseMeta(status, respHeaders)
	}

	var result llm.StreamResult
	if step.response.err != nil {
		result.Err = step.response.err
	}
	result.Messages = append(req.Messages, step.response.resultMessages()...)
	return result.Messages[len(req.Messages):], result.Err
}

// Reset zeroes the call counter so the provider can be reused.
func (m *MockProvider) Reset() {
	m.mu.Lock()
	m.called = 0
	m.mu.Unlock()
}

// Called returns how many script steps have been consumed.
func (m *MockProvider) Called() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.called
}

// ResponseBuilder builds a scripted response via chained method calls.
type ResponseBuilder struct {
	mock    *MockProvider
	matcher Matcher
	r       response
}

// RespondText appends a text response to the script.
func (rb *ResponseBuilder) RespondText(text string) *ResponseBuilder {
	rb.r.events = append(rb.r.events, llm.TextDeltaEvent{Delta: text}, llm.MessageEndEvent{StopReason: llm.StopReasonStop})
	rb.r.resultMsgs = append(rb.r.resultMsgs, llm.Message{
		Role:    "assistant",
		Content: llm.TextContent{Text: text},
	})
	return rb
}

// RespondToolCall appends a tool call response to the script.
func (rb *ResponseBuilder) RespondToolCall(name string, args json.RawMessage) *ResponseBuilder {
	callID := fmt.Sprintf("call_%d", rb.mock.called)
	rb.r.events = append(rb.r.events, llm.ToolCallStartEvent{
		CallID:   callID,
		ToolName: name,
		Args:     args,
	}, llm.ToolCallEndEvent{
		CallID:   callID,
		ToolName: name,
		Args:     args,
	}, llm.MessageEndEvent{StopReason: llm.StopReasonToolUse})
	rb.r.resultMsgs = append(rb.r.resultMsgs, llm.Message{
		Role: "assistant",
		Content: llm.ToolCallContent{
			CallID:   callID,
			ToolName: name,
			Args:     args,
		},
	})
	return rb
}

// Terminate marks the tool call response as terminating the run.
// This is only meaningful when used with RespondToolCall and consumed by pi-core-agent-go.
func (rb *ResponseBuilder) Terminate() *ResponseBuilder {
	// The mock stores termination intent on the last result message.
	// pi-core-agent-go reads Terminate from ToolResult, not from the LLM layer,
	// so this is a no-op at the mock-provider level. The test constructs the
	// ToolResult with Terminate:true directly in the tool executor.
	return rb
}

// RespondThinking appends a thinking response to the script.
func (rb *ResponseBuilder) RespondThinking(text string) *ResponseBuilder {
	// A thinking-only turn with no tool call is a normal completion (both real
	// providers report their native "stop" finish reason here when no tool
	// calls occurred, regardless of whether the turn carried visible text),
	// so StopReasonStop is the faithful reason, not the zero-value Unknown.
	rb.r.events = append(rb.r.events, llm.ThinkingDeltaEvent{Delta: text}, llm.MessageEndEvent{StopReason: llm.StopReasonStop})
	rb.r.resultMsgs = append(rb.r.resultMsgs, llm.Message{
		Role:    "assistant",
		Content: llm.ThinkingContent{Text: text},
	})
	return rb
}

// Error causes the stream to fail with err.
func (rb *ResponseBuilder) Error(err error) *ResponseBuilder {
	rb.r.err = err
	return rb
}

// StreamError emits an LLMErrorEvent during the stream.
func (rb *ResponseBuilder) StreamError(err error) *ResponseBuilder {
	rb.r.streamError = err
	return rb
}

// Status sets the HTTP status code reported via setResponseMeta.
func (rb *ResponseBuilder) Status(code int) *ResponseBuilder {
	rb.r.status = code
	return rb
}

// RespHeaders sets the response headers reported via setResponseMeta.
func (rb *ResponseBuilder) RespHeaders(h map[string]string) *ResponseBuilder {
	rb.r.respHeaders = h
	return rb
}

// Delay causes the mock to wait d before emitting events.
func (rb *ResponseBuilder) Delay(d time.Duration) *ResponseBuilder {
	rb.r.delay = d
	return rb
}

// Add commits the built response to the mock's script.
func (rb *ResponseBuilder) Add() {
	rb.mock.scripts = append(rb.mock.scripts, scriptStep{
		matcher:   rb.matcher,
		response:  rb.r,
		callIndex: len(rb.mock.scripts) + 1,
	})
}

func (r response) resultMessages() []llm.Message {
	if len(r.resultMsgs) == 0 && len(r.events) > 0 {
		return nil
	}
	return r.resultMsgs
}
