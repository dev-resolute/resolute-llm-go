package llm

import (
	"fmt"
	"regexp"
)

// contextOverflowRE matches provider "maximum context length" messages in both the
// "of N tokens"/"is N tokens" form and the parenthesized "(N)" form broadened
// upstream in 0.79.2.
var contextOverflowRE = regexp.MustCompile(`(?i)maximum context length\s*(?:\(\s*\d+\s*\)|(?:is|of)\s+\d+\s+tokens)`)

// AsContextOverflow classifies provider errors: when err's message reports the
// model's maximum context length was exceeded, it returns an error wrapping
// ErrContextOverflow so callers can react via errors.Is (Compact + retry,
// truncate, or switch models). Other errors and nil pass through unchanged.
func AsContextOverflow(err error) error {
	if err == nil {
		return nil
	}
	if contextOverflowRE.MatchString(err.Error()) {
		return fmt.Errorf("%w: %v", ErrContextOverflow, err)
	}
	return err
}
