// Package mock provides a first-class MockProvider for testing code that
// consumes pi-llm-go.
package mock

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"time"

	"github.com/resolute-sh/pi-llm-go"
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
}

// MockProvider is a scripted LLMProvider for tests.
type MockProvider struct {
	name    string
	scripts []scriptStep
	called  int
}

// New creates a fresh MockProvider with the given name.
func New(name string) *MockProvider {
	if name == "" {
		name = "mock"
	}
	return &MockProvider{name: name}
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
	evCh := make(chan llm.LLMEvent, 16)
	doneCh := make(chan llm.StreamResult, 1)

	go func() {
		defer close(evCh)
		defer close(doneCh)

		if m.called >= len(m.scripts) {
			doneCh <- llm.StreamResult{
				Messages: req.Messages,
				Err:      fmt.Errorf("mock: unexpected call #%d (no script steps remaining): %w", m.called+1, llm.ErrProviderFatal),
			}
			return
		}

		step := m.scripts[m.called]
		m.called++

		if !step.matcher.Match(req.Messages) {
			doneCh <- llm.StreamResult{
				Messages: req.Messages,
				Err:      fmt.Errorf("mock: call #%d did not match script step %d: %w", m.called, step.callIndex, llm.ErrProviderFatal),
			}
			return
		}

		if step.response.delay > 0 {
			select {
			case <-time.After(step.response.delay):
			case <-ctx.Done():
				doneCh <- llm.StreamResult{Messages: req.Messages, Err: context.Cause(ctx)}
				return
			}
		}

		for _, ev := range step.response.events {
			select {
			case evCh <- ev:
			case <-ctx.Done():
				doneCh <- llm.StreamResult{Messages: req.Messages, Err: context.Cause(ctx)}
				return
			}
		}

		if step.response.streamError != nil {
			select {
			case evCh <- llm.LLMErrorEvent{Error: step.response.streamError, Transient: false}:
			case <-ctx.Done():
				doneCh <- llm.StreamResult{Messages: req.Messages, Err: context.Cause(ctx)}
				return
			}
		}

		var result llm.StreamResult
		if step.response.err != nil {
			result.Err = step.response.err
		}
		result.Messages = append(req.Messages, step.response.resultMessages()...)
		doneCh <- result
	}()

	return llm.NewEventStream(evCh, doneCh)
}

// Reset zeroes the call counter so the provider can be reused.
func (m *MockProvider) Reset() { m.called = 0 }

// Called returns how many script steps have been consumed.
func (m *MockProvider) Called() int { return m.called }

// ResponseBuilder builds a scripted response via chained method calls.
type ResponseBuilder struct {
	mock    *MockProvider
	matcher Matcher
	r       response
}

// RespondText appends a text response to the script.
func (rb *ResponseBuilder) RespondText(text string) *ResponseBuilder {
	rb.r.events = append(rb.r.events, llm.TextDeltaEvent{Delta: text}, llm.MessageEndEvent{})
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
	}, llm.ToolCallEndEvent{CallID: callID}, llm.MessageEndEvent{})
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

// RespondThinking appends a thinking response to the script.
func (rb *ResponseBuilder) RespondThinking(text string) *ResponseBuilder {
	rb.r.events = append(rb.r.events, llm.ThinkingDeltaEvent{Delta: text}, llm.MessageEndEvent{})
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
