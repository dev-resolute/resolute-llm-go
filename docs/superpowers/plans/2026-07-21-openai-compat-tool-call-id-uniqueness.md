# OpenAI-Compat Tool Call ID Uniqueness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port upstream pi 0.81.0 fix #6854 — keep tool call IDs unique when replaying transcripts into the OpenAI-compatible wire format.

**Architecture:** `toOpenAIMessages` in `openai-compat/provider.go` currently replays `ToolCallContent.CallID` verbatim as both `tool_calls[].id` and `tool_call_id`. Transcripts that cross providers (resolute-harness-go crash-recovery replay) break this: Gemini function calls carry empty IDs (`fc.ID` is often ""), so every call and result gets `""` and pairing is ambiguous — providers 400 or misattribute results. The fix is a small stateful uniquifier applied during conversion: empty IDs get `call_N`, duplicate IDs get an occurrence suffix (`x_2`, `x_3`, …, capped at OpenAI's 40-char limit), and tool results are remapped to their call in occurrence (FIFO) order. Conversion stays deterministic and does not mutate input messages.

**Rejected alternative:** assigning IDs at the source (Gemini provider). Rejected because empty IDs are valid in Gemini-land (Gemini pairs calls by name), the upstream fix lives at the OpenAI conversion boundary, and changing Gemini transcripts would break session interchange.

**Tech Stack:** Go, `github.com/dev-resolute/resolute-llm-go` (`openai-compat` package), stdlib only.

**Repo:** `/Users/maikeffi/playground-ai/pi-research/pi-llm-go`. All commands run from this directory.

**Coding rules:** `docs/go-rules/golang.md` applies — table-driven tests (T-1), `-race` (T-2/G-3), `go vet` gate (G-1), document exported items (API-1). The uniquifier is unexported. Existing code in this package uses `map[string]any` for wire maps; match that style (pre-existing, do not refactor).

---

### Task 1: `toolCallIDUniquifier` — the ID rewriting helper

**Files:**
- Create: `openai-compat/toolcall_id.go`
- Test: `openai-compat/toolcall_id_test.go`

- [ ] **Step 1: Write the failing test**

Create `openai-compat/toolcall_id_test.go`:

```go
package openaicompat

import (
	"reflect"
	"strings"
	"testing"
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./openai-compat/ -run TestToolCallIDUniquifier -v`
Expected: FAIL — `undefined: newToolCallIDUniquifier`

- [ ] **Step 3: Implement the uniquifier**

Create `openai-compat/toolcall_id.go`:

```go
package openaicompat

import "fmt"

// maxToolCallIDLen is OpenAI's tool call id limit; ids longer than this are
// rejected. Mirrors upstream pi-ai's openai-completions converter.
const maxToolCallIDLen = 40

// toolCallIDUniquifier rewrites duplicate or empty tool call IDs during
// OpenAI message conversion so every tool_call on the wire has a distinct id
// and every tool message references the right one.
//
// This matters for cross-provider replay (upstream pi 0.81.0, pi#6854): a
// transcript built by another provider can carry duplicate or empty call IDs
// — Gemini function calls, for example, often have no ID at all. Replayed
// verbatim, every tool_call would carry the same id and call/result pairing
// becomes ambiguous.
//
// Results are matched to calls in occurrence order (FIFO): in a well-formed
// transcript a tool result always follows its call, and when the original IDs
// are identical, order is the only signal available. The mapping is
// deterministic — converting the same transcript twice yields the same ids.
type toolCallIDUniquifier struct {
	counts  map[string]int      // original id -> number of calls seen
	pending map[string][]string // original id -> FIFO queue of final ids awaiting their result
}

func newToolCallIDUniquifier() *toolCallIDUniquifier {
	return &toolCallIDUniquifier{
		counts:  make(map[string]int),
		pending: make(map[string][]string),
	}
}

// call registers a tool call with the given original id and returns the id to
// put on the wire.
func (u *toolCallIDUniquifier) call(id string) string {
	u.counts[id]++
	occurrence := u.counts[id]

	final := id
	switch {
	case id == "":
		final = fmt.Sprintf("call_%d", occurrence)
	case occurrence > 1:
		final = uniquifyToolCallID(id, occurrence)
	}
	u.pending[id] = append(u.pending[id], final)
	return final
}

// result returns the wire id for a tool result whose original id is id:
// the oldest unmatched call with that original id. Orphan results (no
// preceding call) pass through unchanged.
func (u *toolCallIDUniquifier) result(id string) string {
	queue := u.pending[id]
	if len(queue) == 0 {
		return id
	}
	final := queue[0]
	u.pending[id] = queue[1:]
	return final
}

// uniquifyToolCallID suffixes id with its occurrence number, truncating id so
// the result stays within maxToolCallIDLen.
func uniquifyToolCallID(id string, occurrence int) string {
	suffix := fmt.Sprintf("_%d", occurrence)
	if len(id)+len(suffix) <= maxToolCallIDLen {
		return id + suffix
	}
	return id[:maxToolCallIDLen-len(suffix)] + suffix
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./openai-compat/ -run TestToolCallIDUniquifier -v -race`
Expected: PASS — all 6 subtests

- [ ] **Step 5: Commit**

```bash
git add openai-compat/toolcall_id.go openai-compat/toolcall_id_test.go
git commit -m "fix(openai-compat): add tool call ID uniquifier for cross-provider replay"
```

---

### Task 2: Wire the uniquifier into `toOpenAIMessages`

**Files:**
- Modify: `openai-compat/provider.go:242-279` (the `toOpenAIMessages` function)
- Test: `openai-compat/toolcall_id_test.go` (append)

- [ ] **Step 1: Write the failing tests**

Append to `openai-compat/toolcall_id_test.go`:

```go
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
```

Note: `captureBody` and `Config` already exist in `openai-compat/compat_test.go` (same package) — reuse them, do not redefine. Add `"encoding/json"` and `"github.com/dev-resolute/resolute-llm-go"` to the test file's imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./openai-compat/ -run 'TestToOpenAIMessagesToolCallIDUniqueness|TestStreamWireToolCallIDsUnique' -v`
Expected: FAIL — duplicate/empty IDs come back unchanged (`["x","x"]`, `["",""]`)

- [ ] **Step 3: Wire the uniquifier into the conversion**

In `openai-compat/provider.go`, modify `toOpenAIMessages`. Three precise edits:

Edit A — add the uniquifier at the top of the function:

```go
func toOpenAIMessages(messages []llm.Message, compat Compat) []map[string]any {
	ids := newToolCallIDUniquifier()
	var out []map[string]any
```

Edit B — in the `case llm.ToolCallContent:` block, replace `"id":   c.CallID,` with:

```go
		case llm.ToolCallContent:
			callID := ids.call(c.CallID)
			m = map[string]any{
				"role": "assistant",
				"tool_calls": []map[string]any{
					{
						"id":   callID,
```

Edit C — in the `case llm.ToolResultContent:` block, replace `"tool_call_id": c.CallID,` with:

```go
		case llm.ToolResultContent:
			m = map[string]any{
				"role":         "tool",
				"tool_call_id": ids.result(c.CallID),
```

(Only the first line of each case changes plus the `callID` local in Edit B; the rest of each block is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `go test ./openai-compat/ -run 'TestToOpenAIMessagesToolCallIDUniqueness|TestStreamWireToolCallIDsUnique' -v -race`
Expected: PASS — all subtests

- [ ] **Step 5: Run the full package suite and vet**

Run: `go test ./... -race && go vet ./...`
Expected: PASS (no regressions in existing conversion/session/thinking tests)

- [ ] **Step 6: Commit**

```bash
git add openai-compat/provider.go openai-compat/toolcall_id_test.go
git commit -m "fix(openai-compat): keep tool call IDs unique on cross-provider replay (upstream 0.81.0, pi#6854)"
```

---

### Task 3: Docs — CHANGELOG and CONTEXT

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `CONTEXT.md`

- [ ] **Step 1: Add the CHANGELOG entry**

Prepend to `CHANGELOG.md`, after the `# Changelog` heading:

```markdown
## [0.8.2] - 2026-07-21

### Fixed

- **OpenAI-compatible conversion keeps tool call IDs unique on cross-provider
  replay (port of upstream 0.81.0, pi#6854).** `toOpenAIMessages` previously
  replayed `ToolCallContent.CallID` verbatim as both `tool_calls[].id` and the
  matching `tool_call_id`. Transcripts crossing providers — e.g. Gemini
  function calls, whose IDs are empty — produced duplicate or empty IDs on the
  wire, breaking call/result pairing. Conversion now assigns `call_N` to empty
  IDs and suffixes duplicates (`id_2`, `id_3`, …, capped at OpenAI's 40-char
  limit), remapping tool results to their call in occurrence order. Unique
  non-empty IDs pass through unchanged.
```

- [ ] **Step 2: Update the CONTEXT.md glossary**

In `CONTEXT.md`, find the **OpenAI-compatible adapter** entry and append one sentence to its description:

```markdown
Tool call IDs are uniquified at the conversion boundary (empty IDs get `call_N`,
duplicates get occurrence suffixes) so cross-provider transcripts replay with
unambiguous call/result pairing.
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md CONTEXT.md
git commit -m "docs: changelog and glossary for tool call ID uniquification (0.8.2)"
```

---

## Self-Review Notes (completed by plan author)

- **Spec coverage:** Upstream #6854 semantics (unique tool call IDs when calls share a provider call ID) → Tasks 1–2. The upstream pipe-separated Responses-API ID handling is deliberately out of scope (this port has no Responses API target). Char sanitization is out of scope (not needed by any current target; IDs from all supported providers are already wire-safe).
- **Type consistency:** `toolCallIDUniquifier` / `newToolCallIDUniquifier` / `call` / `result` / `uniquifyToolCallID` / `maxToolCallIDLen` used identically across tasks and tests.
- **Placeholders:** none — all code complete.
