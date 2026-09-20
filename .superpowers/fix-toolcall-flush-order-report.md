# Fix: openai-compat tool-call flush order (nondeterministic map iteration)

**Branch:** `fix-toolcall-flush-order` (off `main` @ `9fc06c7`, the v0.10.0 merge commit)
**Commit:** `124645d` — `fix(openai-compat): flush tool calls in first-appearance order`
**Status:** Fix landed, all gates green, branch NOT merged/tagged.

## Finding

`openai-compat/provider.go`'s `readSSE` buffered streaming tool calls in a
`map[string]*toolCallBuffer` (`toolCallBufs`) and, on any non-empty
`finish_reason`, flushed them with `for id, buf := range toolCallBufs`. Go
randomizes map iteration order, so for an assistant turn with multiple tool
calls, the emitted `llm.ToolCallEndEvent` sequence — and the trailing
`resultMessages` (surfaced via `StreamResult.Messages`) — came out in a
different order on different runs. Upstream preserves content order (tool
calls appear in the order the model emitted them), and downstream
(agent-core) now derives execution order from these end events, so
multi-call turns could execute/transcribe out of order.

## Fix

Added a `toolCallOrder []string` alongside `toolCallBufs` in `readSSE`:

- Appended the call ID to `toolCallOrder` at buffer-creation time (the same
  `if !ok { ... }` branch that creates the `toolCallBuffer` and adds it to
  the map), immediately after `toolCallBufs[tc.ID] = buf`.
- The flush block (still gated on any non-empty `finish_reason`, unchanged
  from the LLM-15 behavior) now does `for _, id := range toolCallOrder { buf
  := toolCallBufs[id]; ... }` instead of ranging the map.
- Reset `toolCallOrder = nil` alongside `toolCallBufs = nil` after flush.

Everything else — the "flush on any non-empty finish_reason" behavior, the
finalized `ToolName`/`Args` on `ToolCallEndEvent`, `ToolCallStartEvent`
emission, `StopReason` mapping — is untouched.

### Files changed

- `openai-compat/provider.go` — 6 lines added/changed (`toolCallOrder`
  declaration, append at buffer creation, flush loop rewritten to iterate
  the slice, reset on flush). See diff below.
- `openai-compat/provider_test.go` — new test
  `TestProviderStreamMultipleToolCallsPreserveOrder` (91 lines added).
- `CHANGELOG.md` — new `## [0.10.1] - 2026-07-26` section above `## [0.10.0]
  - 2026-07-25`, one "Fixed" bullet describing the deterministic-ordering
  fix.

```diff
--- a/openai-compat/provider.go
+++ b/openai-compat/provider.go
@@ -362,6 +362,7 @@ func (p *Provider) readSSE(ctx context.Context, resp *http.Response, emit func(l
 	var resultMessages []llm.Message
 	var assistantText strings.Builder
 	var toolCallBufs map[string]*toolCallBuffer
+	var toolCallOrder []string
 	var finishReason string
 	var sawToolCalls bool
 
@@ -400,6 +401,7 @@ func (p *Provider) readSSE(ctx context.Context, resp *http.Response, emit func(l
 				if !ok {
 					buf = &toolCallBuffer{id: tc.ID, name: tc.Function.Name}
 					toolCallBufs[tc.ID] = buf
+					toolCallOrder = append(toolCallOrder, tc.ID)
 					if tc.Function.Name != "" {
 						if err := emit(llm.ToolCallStartEvent{
 							CallID:   tc.ID,
@@ -418,7 +420,8 @@ func (p *Provider) readSSE(ctx context.Context, resp *http.Response, emit func(l
 				finishReason = choice.FinishReason
 			}
 			if choice.FinishReason != "" && toolCallBufs != nil {
-				for id, buf := range toolCallBufs {
+				for _, id := range toolCallOrder {
+					buf := toolCallBufs[id]
 					var args json.RawMessage
 					if buf.args.Len() > 0 {
 						args = json.RawMessage(buf.args.String())
@@ -432,6 +435,7 @@ func (p *Provider) readSSE(ctx context.Context, resp *http.Response, emit func(l
 					})
 				}
 				toolCallBufs = nil
+				toolCallOrder = nil
 				sawToolCalls = true
 			}
 		}
```

## TDD evidence

### Test added

`TestProviderStreamMultipleToolCallsPreserveOrder` in
`openai-compat/provider_test.go` (inserted before
`TestProviderStreamToolCallTruncatedByLength`). Fixture: an SSE stream with
three tool calls — `call_1`/`a`, `call_2`/`b`, `call_3`/`c` — where each
call's name arrives in its own delta chunk (interleaved with the other
calls' name chunks) and each call's arguments arrive in a later, separate
delta chunk, followed by one `finish_reason:"tool_calls"` chunk. Assertions:

1. Exactly 3 `llm.ToolCallEndEvent`s, in the exact sequence
   `call_1/a/{"x":1}`, `call_2/b/{"y":2}`, `call_3/c/{"z":3}`.
2. `StreamResult.Messages` tool-call contents (filtered to
   `llm.ToolCallContent`) are in that same order.

### RED (before the fix — map-range implementation)

Command:

```
env -u GEMINI_API_KEY go test ./openai-compat/ -run TestProviderStreamMultipleToolCallsPreserveOrder -race -count=30 -v
```

Result: **6 of 30 runs FAILED**, 24 passed (probabilistic — Go's map
iteration randomization doesn't guarantee a wrong order every run, but the
large majority of failing seeds/permutations across 30 runs demonstrated the
bug). Example failure output:

```
--- FAIL: TestProviderStreamMultipleToolCallsPreserveOrder (0.00s)
    provider_test.go:221: ToolCallEndEvent[0] = {CallID:call_2 ToolName:b Args:[123 34 121 34 58 50 125] ...}, want CallID=call_1 ToolName=a Args={"x":1}
    provider_test.go:221: ToolCallEndEvent[1] = {CallID:call_3 ToolName:c Args:[123 34 122 34 58 51 125] ...}, want CallID=call_2 ToolName=b Args={"y":2}
    provider_test.go:221: ToolCallEndEvent[2] = {CallID:call_1 ToolName:a Args:[123 34 120 34 58 49 125] ...}, want CallID=call_3 ToolName=c Args={"z":3}
    provider_test.go:236: Messages tool-call[0] = {CallID:call_2 ...} want call_1
    provider_test.go:236: Messages tool-call[1] = {CallID:call_3 ...} want call_2
    provider_test.go:236: Messages tool-call[2] = {CallID:call_1 ...} want call_3
```

Overall: `exit=1`, `grep -c "^--- PASS"` = 24, `grep -c "^--- FAIL"` = 6.

### GREEN (after the fix — toolCallOrder slice)

Same command:

```
env -u GEMINI_API_KEY go test ./openai-compat/ -run TestProviderStreamMultipleToolCallsPreserveOrder -race -count=30 -v
```

Result: `exit=0`, `grep -c "^--- PASS"` = 30, `grep -c "^--- FAIL"` = 0.
**30/30 pass.**

## Gates run before committing

- `gofmt -l .` → empty (no output).
- `go vet ./...` → clean.
- `env -u GEMINI_API_KEY go test -race ./...` → all packages `ok`
  (`resolute-llm-go`, `gemini`, `mock`, `openai-compat`; `openai-compat`
  ran with all local tests including the new one, and no `-race` warnings).
- `GEMINI_API_KEY` was never echoed at any point (only unset via `env -u`
  for hermetic, non-live test runs).

## Scope notes

- Only `readSSE`'s tool-call flush logic was touched; `ToolCallStartEvent`
  emission, the "flush on any non-empty finish_reason" behavior (LLM-15),
  `StopReason`/`mapFinishReason` logic, and all other provider code are
  unchanged.
- Branch was not merged or tagged, per instructions; it sits on
  `fix-toolcall-flush-order`, one commit ahead of the `main` merge commit
  for v0.10.0.
