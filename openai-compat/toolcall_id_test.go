package openaicompat

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/dev-resolute/resolute-llm-go"
)

// runOps feeds a scripted sequence of call/result operations through a
// uniquifier and returns the final IDs it produced, in order.
func runOps(ops ...[2]string) []string {
	u := newToolCallIDUniquifier()
	var got []string
	for _, op := range ops {
		switch op[0] {
		case "call":
			got = append(got, u.call(op[1]))
		case "result":
			got = append(got, u.result(op[1]))
		default:
			panic("bad op: " + op[0])
		}
	}
	return got
}

func TestToolCallIDUniquifier(t *testing.T) {
	long := strings.Repeat("i", maxToolCallIDLen) // exactly 40 chars

	tests := []struct {
		name string
		ops  [][2]string
		want []string
	}{
		{
			name: "unique ids pass through unchanged",
			ops: [][2]string{
				{"call", "a"}, {"result", "a"},
				{"call", "b"}, {"result", "b"},
			},
			want: []string{"a", "a", "b", "b"},
		},
		{
			name: "duplicate ids suffixed per occurrence, results follow in order",
			ops: [][2]string{
				{"call", "x"}, {"call", "x"},
				{"result", "x"}, {"result", "x"},
			},
			want: []string{"x", "x_2", "x", "x_2"},
		},
		{
			name: "empty ids assigned call_N, results follow in order",
			ops: [][2]string{
				{"call", ""}, {"call", ""},
				{"result", ""}, {"result", ""},
			},
			want: []string{"call_1", "call_2", "call_1", "call_2"},
		},
		{
			name: "long duplicate id truncated to 40 chars with suffix",
			ops: [][2]string{
				{"call", long}, {"call", long},
				{"result", long}, {"result", long},
			},
			want: []string{long, long[:maxToolCallIDLen-2] + "_2", long, long[:maxToolCallIDLen-2] + "_2"},
		},
		{
			name: "orphan tool result left untouched",
			ops:  [][2]string{{"result", "ghost"}},
			want: []string{"ghost"},
		},
		{
			name: "third occurrence keeps incrementing",
			ops: [][2]string{
				{"call", "x"}, {"call", "x"}, {"call", "x"},
				{"result", "x"}, {"result", "x"}, {"result", "x"},
			},
			want: []string{"x", "x_2", "x_3", "x", "x_2", "x_3"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := runOps(tt.ops...)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got %v, want %v", got, tt.want)
			}
		})
	}
}

// collectToolIDs extracts tool_call ids and tool result tool_call_ids from
// converted wire messages, in order.
func collectToolIDs(t *testing.T, out []map[string]any) (callIDs, resultIDs []string) {
	t.Helper()
	for _, m := range out {
		switch m["role"] {
		case "assistant":
			tcs, ok := m["tool_calls"].([]map[string]any)
			if !ok {
				continue
			}
			for _, tc := range tcs {
				callIDs = append(callIDs, tc["id"].(string))
			}
		case "tool":
			resultIDs = append(resultIDs, m["tool_call_id"].(string))
		}
	}
	return callIDs, resultIDs
}

func TestToOpenAIMessagesToolCallIDUniqueness(t *testing.T) {
	args := json.RawMessage(`{}`)

	tests := []struct {
		name       string
		messages   []llm.Message
		wantCalls  []string
		wantResult []string
	}{
		{
			name: "unique ids pass through",
			messages: []llm.Message{
				{Role: "assistant", Content: llm.ToolCallContent{CallID: "a", ToolName: "read", Args: args}},
				{Role: "tool", Content: llm.ToolResultContent{CallID: "a", ToolName: "read", Content: "ok"}},
			},
			wantCalls:  []string{"a"},
			wantResult: []string{"a"},
		},
		{
			name: "duplicate ids renamed, results remapped in order",
			messages: []llm.Message{
				{Role: "assistant", Content: llm.ToolCallContent{CallID: "x", ToolName: "read", Args: args}},
				{Role: "assistant", Content: llm.ToolCallContent{CallID: "x", ToolName: "write", Args: args}},
				{Role: "tool", Content: llm.ToolResultContent{CallID: "x", ToolName: "read", Content: "first"}},
				{Role: "tool", Content: llm.ToolResultContent{CallID: "x", ToolName: "write", Content: "second"}},
			},
			wantCalls:  []string{"x", "x_2"},
			wantResult: []string{"x", "x_2"},
		},
		{
			name: "empty ids from gemini replay assigned and paired",
			messages: []llm.Message{
				{Role: "assistant", Content: llm.ToolCallContent{CallID: "", ToolName: "read", Args: args}},
				{Role: "assistant", Content: llm.ToolCallContent{CallID: "", ToolName: "write", Args: args}},
				{Role: "tool", Content: llm.ToolResultContent{CallID: "", ToolName: "read", Content: "first"}},
				{Role: "tool", Content: llm.ToolResultContent{CallID: "", ToolName: "write", Content: "second"}},
			},
			wantCalls:  []string{"call_1", "call_2"},
			wantResult: []string{"call_1", "call_2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := toOpenAIMessages(tt.messages, Compat{})
			calls, results := collectToolIDs(t, out)
			if !reflect.DeepEqual(calls, tt.wantCalls) {
				t.Errorf("call ids: got %v, want %v", calls, tt.wantCalls)
			}
			if !reflect.DeepEqual(results, tt.wantResult) {
				t.Errorf("result ids: got %v, want %v", results, tt.wantResult)
			}
		})
	}
}

// TestStreamWireToolCallIDsUnique pins the end-to-end wire shape: a replayed
// transcript with empty (Gemini-style) call IDs reaches the HTTP body with
// distinct, correctly paired ids.
func TestStreamWireToolCallIDsUnique(t *testing.T) {
	req := llm.LLMRequest{
		Model: "test-model",
		Messages: []llm.Message{
			{Role: "assistant", Content: llm.ToolCallContent{CallID: "", ToolName: "read", Args: json.RawMessage(`{"path":"a"}`)}},
			{Role: "assistant", Content: llm.ToolCallContent{CallID: "", ToolName: "read", Args: json.RawMessage(`{"path":"b"}`)}},
			{Role: "tool", Content: llm.ToolResultContent{CallID: "", ToolName: "read", Content: "a contents"}},
			{Role: "tool", Content: llm.ToolResultContent{CallID: "", ToolName: "read", Content: "b contents"}},
			{Role: "user", Content: llm.TextContent{Text: "go"}},
		},
	}

	captured := captureBody(t, Config{}, req)
	rawMsgs, ok := captured["messages"].([]any)
	if !ok {
		t.Fatalf("captured messages has type %T, want []any", captured["messages"])
	}

	var callIDs, resultIDs []string
	for _, raw := range rawMsgs {
		m, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		switch m["role"] {
		case "assistant":
			tcs, _ := m["tool_calls"].([]any)
			for _, tc := range tcs {
				if tcMap, ok := tc.(map[string]any); ok {
					callIDs = append(callIDs, tcMap["id"].(string))
				}
			}
		case "tool":
			resultIDs = append(resultIDs, m["tool_call_id"].(string))
		}
	}

	want := []string{"call_1", "call_2"}
	if !reflect.DeepEqual(callIDs, want) {
		t.Errorf("wire call ids: got %v, want %v", callIDs, want)
	}
	if !reflect.DeepEqual(resultIDs, want) {
		t.Errorf("wire result ids: got %v, want %v", resultIDs, want)
	}
}
