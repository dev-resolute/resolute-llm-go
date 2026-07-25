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
	"fmt"
	"net/http"
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

	body, err := p.toRequestBody(req)
	if err != nil {
		return nil, fmt.Errorf("openai-compat: building request: %w", err)
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
		transient := isNetworkError(err)
		if emitErr := emit(llm.LLMErrorEvent{Error: err, Transient: transient}); emitErr != nil {
			return nil, emitErr
		}
		return nil, fmt.Errorf("openai-compat: request failed: %w", err)
	}
	defer resp.Body.Close()

	if setResponseMeta != nil {
		respHeaders := make(map[string]string)
		for k, v := range resp.Header {
			if len(v) > 0 {
				respHeaders[k] = v[0]
			}
		}
		setResponseMeta(resp.StatusCode, respHeaders)
	}

	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("openai-compat: HTTP %d", resp.StatusCode)
		transient := resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500
		if emitErr := emit(llm.LLMErrorEvent{Error: err, Transient: transient}); emitErr != nil {
			return nil, emitErr
		}
		return nil, err
	}

	return p.readSSE(ctx, resp, emit)
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
	if len(req.Tools) > 0 {
		body["tools"] = toOpenAITools(req.Tools)
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
	for _, msg := range messages {
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
			m = map[string]any{
				"role":         "tool",
				"tool_call_id": ids.result(c.CallID),
				"content":      c.Content,
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

func toOpenAITools(tools []llm.ToolDef) []map[string]any {
	var out []map[string]any
	for _, tool := range tools {
		out = append(out, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        tool.Name,
				"description": tool.Description,
				"parameters":  json.RawMessage(tool.Schema),
			},
		})
	}
	return out
}

func (p *Provider) readSSE(ctx context.Context, resp *http.Response, emit func(llm.LLMEvent) error) ([]llm.Message, error) {
	scanner := bufio.NewScanner(resp.Body)
	var resultMessages []llm.Message
	var assistantText strings.Builder
	var toolCallBufs map[string]*toolCallBuffer

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

		for _, choice := range chunk.Choices {
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
			if choice.FinishReason == "tool_calls" && toolCallBufs != nil {
				for id, buf := range toolCallBufs {
					var args json.RawMessage
					if buf.args.Len() > 0 {
						args = json.RawMessage(buf.args.String())
					}
					if err := emit(llm.ToolCallEndEvent{CallID: id}); err != nil {
						return nil, err
					}
					resultMessages = append(resultMessages, llm.Message{
						Role: "assistant",
						Content: llm.ToolCallContent{
							CallID:   id,
							ToolName: buf.name,
							Args:     args,
						},
					})
				}
				toolCallBufs = nil
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("openai-compat: SSE read error: %w", err)
	}

	if assistantText.Len() > 0 {
		resultMessages = append(resultMessages, llm.Message{
			Role:    "assistant",
			Content: llm.TextContent{Text: assistantText.String()},
		})
	}

	if err := emit(llm.MessageEndEvent{}); err != nil {
		return nil, err
	}

	return resultMessages, nil
}

type toolCallBuffer struct {
	id   string
	name string
	args strings.Builder
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
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

func isNetworkError(err error) bool {
	if err == nil {
		return false
	}
	return true // conservative: treat all network errors as transient
}
