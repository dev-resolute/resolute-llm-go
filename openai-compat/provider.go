// Package openaicompat provides an LLMProvider implementation that targets
// any OpenAI-compatible HTTP endpoint, including OpenAI, Fireworks, Ollama,
// vLLM, llama.cpp server, and LM Studio.
package openaicompat

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/resolute-sh/pi-llm-go"
)

// Config holds the OpenAI-compatible provider configuration.
type Config struct {
	APIKey    string
	BaseURL   string
	Retry     llm.RetryPolicy
	Headers   map[string]string
}

// Provider implements llm.LLMProvider for OpenAI-compatible endpoints.
type Provider struct {
	client *http.Client
	name   string
	config Config
}

// New creates a Provider from the given Config.
func New(cfg Config) (llm.LLMProvider, error) {
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("openai-compat: BaseURL is required: %w", llm.ErrInvalidModel)
	}
	client := &http.Client{Timeout: 120 * time.Second}
	name := "openai-compat"
	return &Provider{client: client, name: name, config: cfg}, nil
}

// Name implements llm.LLMProvider.
func (p *Provider) Name() string { return p.name }

// Capabilities implements llm.LLMProvider.
func (p *Provider) Capabilities(model string) llm.ProviderCapabilities {
	caps := llm.ProviderCapabilities{
		Streaming:        true,
		ToolCalling:      true,
		PromptCaching:    false,
		Vision:           strings.Contains(model, "vision"),
	}
	if strings.HasPrefix(model, "o") || (strings.HasPrefix(model, "accounts/") && strings.Contains(model, "o")) {
		caps.Thinking = true
	}
	caps.ParallelToolCalls = true
	return caps
}

// Stream implements llm.LLMProvider.
func (p *Provider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	evCh := make(chan llm.LLMEvent, 16)
	doneCh := make(chan llm.StreamResult, 1)

	go func() {
		defer close(evCh)
		defer close(doneCh)

		body, err := p.toRequestBody(req)
		if err != nil {
			doneCh <- llm.StreamResult{Err: fmt.Errorf("openai-compat: building request: %w", err)}
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", p.config.BaseURL+"/chat/completions", bytes.NewReader(body))
		if err != nil {
			doneCh <- llm.StreamResult{Err: fmt.Errorf("openai-compat: creating request: %w", err)}
			return
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")
		if p.config.APIKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+p.config.APIKey)
		}
		for k, v := range p.config.Headers {
			httpReq.Header.Set(k, v)
		}

		resp, err := p.client.Do(httpReq)
		if err != nil {
			transient := isNetworkError(err)
			select {
			case evCh <- llm.LLMErrorEvent{Error: err, Transient: transient}:
			case <-ctx.Done():
			}
			doneCh <- llm.StreamResult{Err: fmt.Errorf("openai-compat: request failed: %w", err)}
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			err := fmt.Errorf("openai-compat: HTTP %d", resp.StatusCode)
			transient := resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500
			select {
			case evCh <- llm.LLMErrorEvent{Error: err, Transient: transient}:
			case <-ctx.Done():
			}
			doneCh <- llm.StreamResult{Err: err}
			return
		}

		resultMessages, err := p.readSSE(ctx, resp, evCh)
		if err != nil {
			doneCh <- llm.StreamResult{Err: err}
			return
		}

		doneCh <- llm.StreamResult{Messages: append(req.Messages, resultMessages...)}
	}()

	return llm.NewEventStream(evCh, doneCh)
}

func (p *Provider) toRequestBody(req llm.LLMRequest) ([]byte, error) {
	body := map[string]any{
		"model":    req.Model,
		"stream":   true,
		"messages": toOpenAIMessages(req.Messages),
	}
	if len(req.Tools) > 0 {
		body["tools"] = toOpenAITools(req.Tools)
	}
	if req.Thinking != llm.ThinkingOff {
		effort := map[llm.ThinkingLevel]string{
			llm.ThinkingLow:    "low",
			llm.ThinkingMedium: "medium",
			llm.ThinkingHigh:   "high",
		}[req.Thinking]
		if req.ProviderHints.OpenAI != nil && req.ProviderHints.OpenAI.ReasoningEffort != "" {
			effort = req.ProviderHints.OpenAI.ReasoningEffort
		}
		if effort != "" {
			body["reasoning_effort"] = effort
		}
	}
	return json.Marshal(body)
}

func toOpenAIMessages(messages []llm.Message) []map[string]any {
	var out []map[string]any
	for _, msg := range messages {
		switch c := msg.Content.(type) {
		case llm.TextContent:
			out = append(out, map[string]any{
				"role":    msg.Role,
				"content": c.Text,
			})
		case llm.ToolCallContent:
			out = append(out, map[string]any{
				"role": "assistant",
				"tool_calls": []map[string]any{
					{
						"id":   c.CallID,
						"type": "function",
						"function": map[string]any{
							"name":      c.ToolName,
							"arguments": string(c.Args),
						},
					},
				},
			})
		case llm.ToolResultContent:
			out = append(out, map[string]any{
				"role":       "tool",
				"tool_call_id": c.CallID,
				"content":    c.Content,
			})
		case llm.ThinkingContent:
			out = append(out, map[string]any{
				"role":    msg.Role,
				"content": c.Text,
			})
		default:
			out = append(out, map[string]any{
				"role":    msg.Role,
				"content": "",
			})
		}
	}
	return out
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

func (p *Provider) readSSE(ctx context.Context, resp *http.Response, evCh chan<- llm.LLMEvent) ([]llm.Message, error) {
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
				select {
				case evCh <- llm.TextDeltaEvent{Delta: choice.Delta.Content}:
				case <-ctx.Done():
					return nil, context.Cause(ctx)
				}
			}
			if choice.Delta.ReasoningContent != "" {
				select {
				case evCh <- llm.ThinkingDeltaEvent{Delta: choice.Delta.ReasoningContent}:
				case <-ctx.Done():
					return nil, context.Cause(ctx)
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
						select {
						case evCh <- llm.ToolCallStartEvent{
							CallID:   tc.ID,
							ToolName: tc.Function.Name,
							Args:     nil,
						}:
						case <-ctx.Done():
							return nil, context.Cause(ctx)
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
					select {
					case evCh <- llm.ToolCallEndEvent{CallID: id}:
					case <-ctx.Done():
						return nil, context.Cause(ctx)
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

	select {
	case evCh <- llm.MessageEndEvent{}:
	case <-ctx.Done():
		return nil, context.Cause(ctx)
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
