package llm

import (
	"testing"
	"time"
)

func TestEventStreamBufferSize(t *testing.T) {
	evCh := make(chan LLMEvent, 4)
	doneCh := make(chan StreamResult, 1)

	s := NewEventStream(evCh, doneCh)

	if cap(evCh) != 4 {
		t.Fatalf("expected buffer size 4, got %d", cap(evCh))
	}
	_ = s
}

func TestLLMEventVariants(t *testing.T) {
	// Verify each concrete variant implements LLMEvent.
	variants := []LLMEvent{
		TextDeltaEvent{Delta: "hello"},
		ThinkingDeltaEvent{Delta: "thinking"},
		ToolCallStartEvent{CallID: "1", ToolName: "foo", Args: []byte(`{}`)},
		ToolCallEndEvent{CallID: "1"},
		MessageEndEvent{},
		LLMErrorEvent{Error: nil, Transient: true},
		LLMRetryEvent{Provider: "mock", Model: "test", Attempt: 1, NextDelay: time.Second, Reason: "rate limit", ServerHint: true},
		UsageEvent{InputTokens: 10, OutputTokens: 5},
	}

	for _, v := range variants {
		if v == nil {
			t.Fatal("nil variant")
		}
	}
}
