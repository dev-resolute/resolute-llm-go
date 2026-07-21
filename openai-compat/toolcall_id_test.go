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
