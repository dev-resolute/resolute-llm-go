// Package openaicompat provides an LLMProvider implementation that targets
// any OpenAI-compatible HTTP endpoint, including OpenAI, Fireworks, Ollama,
// vLLM, llama.cpp server, and LM Studio.
package openaicompat

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/dev-resolute/resolute-llm-go"
)

// Config holds the OpenAI-compatible provider configuration.
type Config struct {
	// Name is the provider's identifier, used in model references like
	// "<name>/<model-id>" and as the registry key. Distinct OpenAI-compatible
	// targets (e.g. "xai", "mistral") must each set a distinct Name.
	Name      string
	APIKey    string
	GetAPIKey func(ctx context.Context) (string, error)
	BaseURL   string
	Retry     llm.RetryPolicy
	Headers   map[string]string
	Compat    Compat
	// SupportsStrictTools overrides whether function tools on this instance may
	// request provider-side "strict" JSON-schema-enforced sampling. nil means
	// omitted: the classifier default applies (true for a plain instance and
	// every current named family — see classification.strictTools). A non-nil
	// pointer — including one pointing at false — always wins over the
	// classifier.
	SupportsStrictTools *bool
}

// Provider implements llm.LLMProvider for OpenAI-compatible endpoints.
type Provider struct {
	client *http.Client
	name   string
	config Config
	// classify resolves per-model behaviour catalog-free for the built-in provider
	// families (xAI, Mistral, …). It is nil for a plain New, which keeps the coarse
	// prefix heuristic; the family constructors set it.
	classify func(model string) classification
}

// New creates a Provider from the given Config.
func New(cfg Config) (llm.LLMProvider, error) {
	return newProvider(cfg)
}

func newProvider(cfg Config) (*Provider, error) {
	if cfg.Name == "" {
		return nil, fmt.Errorf("openai-compat: Name is required: %w", llm.ErrInvalidModel)
	}
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("openai-compat: BaseURL is required: %w", llm.ErrInvalidModel)
	}
	client := &http.Client{Timeout: 120 * time.Second}
	return &Provider{client: client, name: cfg.Name, config: cfg}, nil
}

// Name implements llm.LLMProvider.
func (p *Provider) Name() string { return p.name }

// Capabilities implements llm.LLMProvider.
func (p *Provider) Capabilities(model string) llm.ProviderCapabilities {
	if p.classify != nil {
		c := p.classify(model)
		return llm.ProviderCapabilities{
			Streaming:         true,
			ToolCalling:       true,
			ParallelToolCalls: true,
			Thinking:          c.thinking,
			Vision:            c.vision,
		}
	}
	caps := llm.ProviderCapabilities{
		Streaming:     true,
		ToolCalling:   true,
		PromptCaching: false,
		Vision:        strings.Contains(model, "vision"),
	}
	if strings.HasPrefix(model, "o") || (strings.HasPrefix(model, "accounts/") && strings.Contains(model, "o")) {
		caps.Thinking = true
	}
	if p.config.Compat.ThinkingFormat != ThinkingReasoningEffort {
		caps.Thinking = true
	}
	caps.ParallelToolCalls = true
	return caps
}

// Stream implements llm.LLMProvider.
func (p *Provider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	return llm.Run(ctx, req, p.produce)
}

func (p *Provider) produce(ctx context.Context, req llm.LLMRequest, emit func(llm.LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]llm.Message, error) {
	if req.Transport == llm.TransportWebSocket {
		return nil, fmt.Errorf("openai-compat: %w: %s", llm.ErrTransportUnsupported, req.Transport)
	}

	body, err := p.toRequestBody(req)
	if err != nil {
		// Every error toRequestBody can produce today is deterministic (a
		// strict-sampling resolution failure — require on an unsupported
		// instance, or an invalid Strict value — or malformed tool.Schema
		// rejected by json.Marshal): it will fail identically on every retry,
		// so it is fatal and issued before any HTTP request.
		fatalErr := fmt.Errorf("%w: openai-compat: building request: %w", llm.ErrProviderFatal, err)
		if emitErr := emit(llm.LLMErrorEvent{Error: fatalErr, Transient: false}); emitErr != nil {
			return nil, emitErr
		}
		return nil, fatalErr
	}

	// The retried boundary is the stream open: classification failures inside
	// open emit nothing, the ladder emits LLMRetryEvent per attempt, and only
	// the final failure becomes an LLMErrorEvent below. Streaming proper
	// (readSSE) runs unretried so content is never duplicated.
	var resp *http.Response
	err = llm.Retry(ctx, p.config.Retry, p.name, req.Model, emit, func(ctx context.Context) error {
		var openErr error
		resp, openErr = p.open(ctx, req, body, headers, setResponseMeta)
		return openErr
	})
	if err != nil {
		var terr *llm.TransientError
		if emitErr := emit(llm.LLMErrorEvent{Error: err, Transient: errors.As(err, &terr)}); emitErr != nil {
			return nil, emitErr
		}
		return nil, err
	}
	defer resp.Body.Close()

	return p.readSSE(ctx, resp, emit)
}

// open performs one stream-open attempt: resolves the API key (per attempt, so
// expiring credentials refresh across retries), issues the HTTP request, and
// classifies failures for the retry ladder — transport errors, 408/409/429,
// and 5xx are transient (upstream isRetryableProviderError, with the
// x-should-retry header veto/override), everything else fatal.
func (p *Provider) open(ctx context.Context, req llm.LLMRequest, body []byte, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) (*http.Response, error) {
	apiKey := p.config.APIKey
	if p.config.GetAPIKey != nil {
		key, err := p.config.GetAPIKey(ctx)
		if err != nil {
			return nil, fmt.Errorf("openai-compat: refreshing api key: %w", err)
		}
		if key != "" {
			apiKey = key
		}
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai-compat: creating request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	}
	if req.SessionID != "" {
		httpReq.Header.Set("session_id", req.SessionID)
		httpReq.Header.Set("x-client-request-id", req.SessionID)
		httpReq.Header.Set("x-session-affinity", req.SessionID)
	}
	for k, v := range p.config.Headers {
		httpReq.Header.Set(k, v)
	}
	for k, v := range headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		// No status at all: a transport failure, always transient.
		return nil, &llm.TransientError{Err: fmt.Errorf("openai-compat: request failed: %w", err)}
	}

	if setResponseMeta != nil {
		respHeaders := make(map[string]string)
		for k, v := range resp.Header {
			if len(v) > 0 {
				respHeaders[k] = v[0]
			}
		}
		setResponseMeta(resp.StatusCode, respHeaders)
	}

	if resp.StatusCode == http.StatusOK {
		return resp, nil
	}
	defer resp.Body.Close()

	httpErr := fmt.Errorf("openai-compat: HTTP %d", resp.StatusCode)
	switch resp.Header.Get("x-should-retry") {
	case "true":
		return nil, &llm.TransientError{Err: httpErr, RetryAfter: parseRetryAfter(resp.Header)}
	case "false":
		return nil, httpErr
	}
	if resp.StatusCode == http.StatusRequestTimeout ||
		resp.StatusCode == http.StatusConflict ||
		resp.StatusCode == http.StatusTooManyRequests ||
		resp.StatusCode >= 500 {
		return nil, &llm.TransientError{Err: httpErr, RetryAfter: parseRetryAfter(resp.Header)}
	}
	return nil, httpErr
}

// parseRetryAfter reads the server's requested wait: retry-after-ms
// (milliseconds) wins over retry-after (seconds or HTTP-date). 0 when absent
// or unparsable; may be negative for a past HTTP-date, which the ladder's
// sleep clamps.
func parseRetryAfter(h http.Header) time.Duration {
	if v := h.Get("retry-after-ms"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return time.Duration(f * float64(time.Millisecond))
		}
	}
	if v := h.Get("retry-after"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return time.Duration(f * float64(time.Second))
		}
		if ts, err := http.ParseTime(v); err == nil {
			return time.Until(ts)
		}
	}
	return 0
}

// maxPromptCacheKeyLength is the OpenAI prompt_cache_key limit, matching upstream Pi.
const maxPromptCacheKeyLength = 64

// clampPromptCacheKey truncates the key to maxPromptCacheKeyLength runes.
func clampPromptCacheKey(key string) string {
	runes := []rune(key)
	if len(runes) <= maxPromptCacheKeyLength {
		return key
	}
	return string(runes[:maxPromptCacheKeyLength])
}

func (p *Provider) toRequestBody(req llm.LLMRequest) ([]byte, error) {
	body := map[string]any{
		"model":    req.Model,
		"stream":   true,
		"messages": toOpenAIMessages(req.Messages, p.config.Compat),
	}
	if p.supportsUsageInStreaming() {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	if len(req.Tools) > 0 {
		tools, err := toOpenAITools(req.Tools, p.supportsStrictTools(req.Model))
		if err != nil {
			return nil, err
		}
		body["tools"] = tools
	}
	if req.SessionID != "" {
		body["prompt_cache_key"] = clampPromptCacheKey(req.SessionID)
	}
	p.applyThinking(body, req)
	if p.config.Compat.MaxTokens > 0 {
		body["max_tokens"] = p.config.Compat.MaxTokens
	}
	return json.Marshal(body)
}

func (p *Provider) applyThinking(body map[string]any, req llm.LLMRequest) {
	compat := p.config.Compat
	if compat.ThinkingFormat == ThinkingDeepSeek {
		if req.Thinking != llm.ThinkingOff {
			body["thinking"] = map[string]any{"type": "enabled"}
			if compat.SupportsReasoningEffort {
				if e := reasoningEffort(req); e != "" {
					body["reasoning_effort"] = e
				}
			}
		} else {
			body["thinking"] = map[string]any{"type": "disabled"}
		}
		return
	}
	if compat.ThinkingFormat == ThinkingChatTemplate {
		body["chat_template_kwargs"] = map[string]any{"enable_thinking": req.Thinking != llm.ThinkingOff}
		return
	}
	if compat.ThinkingFormat == ThinkingQwen {
		body["enable_thinking"] = req.Thinking != llm.ThinkingOff
		return
	}
	if req.Thinking != llm.ThinkingOff && p.acceptsReasoningEffort(req.Model) {
		if e := reasoningEffort(req); e != "" {
			body["reasoning_effort"] = e
		}
	}
}

// acceptsReasoningEffort reports whether reasoning_effort may be sent for this
// model. A plain provider always sends it (unchanged behaviour); a family with a
// classifier omits it for models that reason but reject the param (xAI grok-4,
// Mistral Magistral), which would otherwise return HTTP 400.
func (p *Provider) acceptsReasoningEffort(model string) bool {
	if p.classify == nil {
		return true
	}
	return p.classify(model).reasoningEffort
}

// supportsStrictTools reports whether function tools on this instance/model may
// request provider-side "strict" JSON-schema-enforced sampling.
// Config.SupportsStrictTools, when set, always wins; otherwise the classifier
// default applies — true for a plain instance (no classifier) and every
// current named family (see classification.strictTools for the future
// per-family off-switch).
func (p *Provider) supportsStrictTools(model string) bool {
	if p.config.SupportsStrictTools != nil {
		return *p.config.SupportsStrictTools
	}
	if p.classify == nil {
		return true
	}
	return p.classify(model).strictTools
}

// imageURLPart renders an ImageContent as an OpenAI image_url content part
// with an inline data URL.
func imageURLPart(img llm.ImageContent) map[string]any {
	return map[string]any{
		"type": "image_url",
		"image_url": map[string]any{
			"url": "data:" + img.MimeType + ";base64," + base64.StdEncoding.EncodeToString(img.Data),
		},
	}
}

func toOpenAIMessages(messages []llm.Message, compat Compat) []map[string]any {
	ids := newToolCallIDUniquifier()
	var out []map[string]any
	var pendingImages []map[string]any
	flushToolImages := func() {
		if len(pendingImages) == 0 {
			return
		}
		// Leading text part matches upstream openai-completions.ts:1244-1252,
		// which prepends a fixed marker before the batched image_url parts.
		content := append([]map[string]any{
			{"type": "text", "text": "Attached image(s) from tool result:"},
		}, pendingImages...)
		out = append(out, map[string]any{"role": "user", "content": content})
		pendingImages = nil
	}
	for _, msg := range messages {
		if _, isToolResult := msg.Content.(llm.ToolResultContent); !isToolResult {
			flushToolImages()
		}
		var m map[string]any
		switch c := msg.Content.(type) {
		case llm.TextContent:
			m = map[string]any{"role": msg.Role, "content": c.Text}
		case llm.ToolCallContent:
			callID := ids.call(c.CallID)
			m = map[string]any{
				"role": "assistant",
				"tool_calls": []map[string]any{
					{
						"id":   callID,
						"type": "function",
						"function": map[string]any{
							"name":      c.ToolName,
							"arguments": string(c.Args),
						},
					},
				},
			}
		case llm.ToolResultContent:
			// Empty tool text becomes a placeholder; images from a run of
			// consecutive tool results are hoisted and batched into one
			// trailing user message (upstream 0.82.0 behaviour). The
			// placeholder is three-way: upstream openai-completions.ts
			// (`toolResultText = hasText ? textResult : hasImages ?
			// "(see attached image)" : "(no tool output)"`).
			text := c.Content
			if text == "" {
				if len(c.Images) > 0 {
					text = "(see attached image)"
				} else {
					text = "(no tool output)"
				}
			}
			m = map[string]any{
				"role":         "tool",
				"tool_call_id": ids.result(c.CallID),
				"content":      text,
			}
			for _, img := range c.Images {
				pendingImages = append(pendingImages, imageURLPart(img))
			}
		case llm.ThinkingContent:
			m = map[string]any{"role": msg.Role, "content": c.Text}
		case llm.ImageContent:
			m = map[string]any{
				"role":    msg.Role,
				"content": []map[string]any{imageURLPart(c)},
			}
		default:
			m = map[string]any{"role": msg.Role, "content": ""}
		}
		ensureReasoningContent(m, compat)
		out = append(out, m)
	}
	flushToolImages()
	return out
}

// ensureReasoningContent adds an empty reasoning_content field to assistant
// messages when the model requires its presence (e.g. DeepSeek).
func ensureReasoningContent(m map[string]any, compat Compat) {
	if !compat.RequiresReasoningContentOnAssistantMessages || m["role"] != "assistant" {
		return
	}
	if _, ok := m["reasoning_content"]; !ok {
		m["reasoning_content"] = ""
	}
}

// toOpenAITools converts tool defs to the OpenAI-compatible wire shape. When
// strictSupported, every function tool object carries "strict": true when
// llm.ResolveStrictSampling resolves the tool strict, false otherwise
// (upstream convertTools parity, openai-completions.ts:1301-1311). When
// !strictSupported, the "strict" key is omitted entirely from every tool —
// some providers 400 on unrecognized fields. An error return means a tool
// requires strict sampling the instance can't honor, or carries an invalid
// Strict value; the caller must treat this as a fatal pre-flight error.
func toOpenAITools(tools []llm.ToolDef, strictSupported bool) ([]map[string]any, error) {
	var out []map[string]any
	for _, tool := range tools {
		strict, err := llm.ResolveStrictSampling(tool, strictSupported)
		if err != nil {
			return nil, err
		}
		fn := map[string]any{
			"name":        tool.Name,
			"description": tool.Description,
			"parameters":  json.RawMessage(tool.Schema),
		}
		if strictSupported {
			fn["strict"] = strict
		}
		out = append(out, map[string]any{
			"type":     "function",
			"function": fn,
		})
	}
	return out, nil
}

func (p *Provider) readSSE(ctx context.Context, resp *http.Response, emit func(llm.LLMEvent) error) ([]llm.Message, error) {
	scanner := bufio.NewScanner(resp.Body)
	var resultMessages []llm.Message
	var assistantText strings.Builder
	var toolCallBufs map[string]*toolCallBuffer
	var toolCallOrder []string
	var finishReason string
	var sawToolCalls bool
	var lastUsage *usageChunk

	// flushToolCalls ends every buffered call in first-appearance order and
	// moves it into the result messages. Calls flush when a finish_reason
	// chunk arrives, or at stream end for providers that omit finish_reason.
	flushToolCalls := func() error {
		for _, id := range toolCallOrder {
			buf := toolCallBufs[id]
			var args json.RawMessage
			if buf.args.Len() > 0 {
				args = json.RawMessage(buf.args.String())
			}
			if err := emit(llm.ToolCallEndEvent{CallID: id, ToolName: buf.name, Args: args}); err != nil {
				return err
			}
			resultMessages = append(resultMessages, llm.Message{
				Role:    "assistant",
				Content: llm.ToolCallContent{CallID: id, ToolName: buf.name, Args: args},
			})
		}
		if toolCallBufs != nil {
			sawToolCalls = true
		}
		toolCallBufs = nil
		toolCallOrder = nil
		return nil
	}

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		// Usage rides its own chunk (possibly with empty choices), or
		// Moonshot-style on a choice when the chunk has none. Last report wins.
		if chunk.Usage != nil {
			lastUsage = chunk.Usage
		}

		for _, choice := range chunk.Choices {
			if chunk.Usage == nil && choice.Usage != nil {
				lastUsage = choice.Usage
			}
			if choice.Delta.Content != "" {
				assistantText.WriteString(choice.Delta.Content)
				if err := emit(llm.TextDeltaEvent{Delta: choice.Delta.Content}); err != nil {
					return nil, err
				}
			}
			if choice.Delta.ReasoningContent != "" {
				if err := emit(llm.ThinkingDeltaEvent{Delta: choice.Delta.ReasoningContent}); err != nil {
					return nil, err
				}
			}
			for _, tc := range choice.Delta.ToolCalls {
				if toolCallBufs == nil {
					toolCallBufs = make(map[string]*toolCallBuffer)
				}
				buf, ok := toolCallBufs[tc.ID]
				if !ok {
					buf = &toolCallBuffer{id: tc.ID, name: tc.Function.Name}
					toolCallBufs[tc.ID] = buf
					toolCallOrder = append(toolCallOrder, tc.ID)
					if tc.Function.Name != "" {
						if err := emit(llm.ToolCallStartEvent{
							CallID:   tc.ID,
							ToolName: tc.Function.Name,
							Args:     nil,
						}); err != nil {
							return nil, err
						}
					}
				}
				if tc.Function.Arguments != "" {
					buf.args.WriteString(tc.Function.Arguments)
				}
			}
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
				if err := flushToolCalls(); err != nil {
					return nil, err
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("openai-compat: SSE read error: %w", err)
	}

	// A provider expected to send finish_reason that ends without one is a
	// protocol error (upstream's pending invariant), not a silent unknown stop.
	supportsFinishReason := p.supportsFinishReason()
	if finishReason == "" && supportsFinishReason {
		err := fmt.Errorf("openai-compat: %w: stream ended without finish_reason", llm.ErrMalformedResponse)
		if emitErr := emit(llm.LLMErrorEvent{Error: err, Transient: false}); emitErr != nil {
			return nil, emitErr
		}
		return nil, err
	}
	// Providers known to omit finish_reason end here with calls still
	// buffered: flush them so the inference below sees the full content.
	if err := flushToolCalls(); err != nil {
		return nil, err
	}

	if assistantText.Len() > 0 {
		resultMessages = append(resultMessages, llm.Message{
			Role:    "assistant",
			Content: llm.TextContent{Text: assistantText.String()},
		})
	}

	stopReason, err := compatStopReason(finishReason, sawToolCalls, supportsFinishReason)
	if err != nil {
		if emitErr := emit(llm.LLMErrorEvent{Error: err, Transient: false}); emitErr != nil {
			return nil, emitErr
		}
		return nil, err
	}

	// At most one UsageEvent per stream, carrying the request's final totals,
	// so consumer-side accumulation is exactly-once per call.
	if lastUsage != nil {
		if err := emit(mapUsageChunk(lastUsage)); err != nil {
			return nil, err
		}
	}

	if err := emit(llm.MessageEndEvent{StopReason: stopReason}); err != nil {
		return nil, err
	}

	return resultMessages, nil
}

// supportsUsageInStreaming resolves Compat.SupportsUsageInStreaming: nil
// means the provider accepts stream_options.include_usage (upstream default).
func (p *Provider) supportsUsageInStreaming() bool {
	return p.config.Compat.SupportsUsageInStreaming == nil || *p.config.Compat.SupportsUsageInStreaming
}

// supportsFinishReason resolves Compat.SupportsFinishReason: nil means the
// provider is expected to terminate streams with finish_reason (upstream
// default true).
func (p *Provider) supportsFinishReason() bool {
	return p.config.Compat.SupportsFinishReason == nil || *p.config.Compat.SupportsFinishReason
}

// compatStopReason maps an OpenAI finish_reason to the portable StopReason, or
// to a fatal error for terminal reasons without a portable mapping (upstream
// #7272). An empty reason is a protocol error for providers expected to send
// one, and inferred from content for providers known to omit it. Length wins
// over tool calls: truncated arguments may be incomplete (upstream #6285) —
// and an error stop's calls may be equally borked, so error reasons win over
// toolUse too (upstream maps any reason with a toolCall block to toolUse,
// silently bypassing both guards).
func compatStopReason(finishReason string, sawToolCalls, supportsFinishReason bool) (llm.StopReason, error) {
	if finishReason == "" {
		if supportsFinishReason {
			return llm.StopReasonUnknown, fmt.Errorf("openai-compat: %w: stream ended without finish_reason", llm.ErrMalformedResponse)
		}
		if sawToolCalls {
			return llm.StopReasonToolUse, nil
		}
		return llm.StopReasonStop, nil
	}
	switch finishReason {
	case "length":
		return llm.StopReasonLength, nil
	case "tool_calls", "function_call":
		return llm.StopReasonToolUse, nil
	case "stop", "end":
		// Some servers report "stop" despite having streamed tool calls.
		if sawToolCalls {
			return llm.StopReasonToolUse, nil
		}
		return llm.StopReasonStop, nil
	default:
		return llm.StopReasonUnknown, fmt.Errorf("openai-compat: %w: finish_reason %q", llm.ErrProviderStop, finishReason)
	}
}

type toolCallBuffer struct {
	id   string
	name string
	args strings.Builder
}

type usagePromptDetails struct {
	CachedTokens     int `json:"cached_tokens"`
	CacheWriteTokens int `json:"cache_write_tokens"`
}

// usageChunk is one usage report on a stream chunk (or, for Moonshot-style
// servers, on a choice). prompt_cache_hit_tokens is the legacy DeepSeek form
// of cached_tokens.
type usageChunk struct {
	PromptTokens         int                 `json:"prompt_tokens"`
	CompletionTokens     int                 `json:"completion_tokens"`
	PromptCacheHitTokens int                 `json:"prompt_cache_hit_tokens"`
	PromptTokensDetails  *usagePromptDetails `json:"prompt_tokens_details"`
}

// mapUsageChunk computes the portable totals for one request (upstream
// parseChunkUsage): input = prompt − cache-read − cache-write, floored at
// zero; output = completion (already includes reasoning tokens).
func mapUsageChunk(u *usageChunk) llm.UsageEvent {
	cached := u.PromptCacheHitTokens
	var cacheWrite int
	if u.PromptTokensDetails != nil {
		cached += u.PromptTokensDetails.CachedTokens
		cacheWrite = u.PromptTokensDetails.CacheWriteTokens
	}
	input := u.PromptTokens - cached - cacheWrite
	if input < 0 {
		input = 0
	}
	return llm.UsageEvent{InputTokens: input, OutputTokens: u.CompletionTokens}
}

type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
			ToolCalls        []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string      `json:"finish_reason"`
		Usage        *usageChunk `json:"usage"`
	} `json:"choices"`
	Usage *usageChunk `json:"usage"`
}
