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
