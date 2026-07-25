// message_image_test.go
package llm

import (
	"encoding/json"
	"testing"
)

// ImageContent must be a Content and round-trip through JSON with base64 data.
func TestImageContentJSONRoundTrip(t *testing.T) {
	var _ Content = ImageContent{} // compile-time: sealed interface membership

	in := ImageContent{Data: []byte{0x89, 0x50, 0x4e, 0x47}, MimeType: "image/png"}
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// encoding/json encodes []byte as base64: 0x89504e47 -> "iVBORw=="
	want := `{"Data":"iVBORw==","MimeType":"image/png"}`
	if string(raw) != want {
		t.Errorf("marshal = %s, want %s", raw, want)
	}
	var out ImageContent
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if string(out.Data) != string(in.Data) || out.MimeType != in.MimeType {
		t.Errorf("round-trip = %+v, want %+v", out, in)
	}
}

func TestToolResultContentImagesField(t *testing.T) {
	c := ToolResultContent{
		CallID:  "call_1",
		Content: "Read image file [image/png]",
		Images:  []ImageContent{{Data: []byte{1}, MimeType: "image/png"}},
	}
	if len(c.Images) != 1 {
		t.Fatalf("Images len = %d, want 1", len(c.Images))
	}
}
