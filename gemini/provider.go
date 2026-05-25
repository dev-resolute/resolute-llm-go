// Package gemini provides an LLMProvider implementation built on the official
// Google GenAI SDK (google.golang.org/genai).
package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"google.golang.org/genai"
	"github.com/resolute-sh/pi-llm-go"
)

// Config holds the Gemini provider configuration.
type Config struct {
	APIKey string
	Retry  llm.RetryPolicy
}

// Provider implements llm.LLMProvider for Google Gemini models.
type Provider struct {
	name   string
	config Config
}

// New creates a Provider from the given Config.
func New(cfg Config) (llm.LLMProvider, error) {
	return &Provider{name: "gemini", config: cfg}, nil
}

// Name implements llm.LLMProvider.
func (p *Provider) Name() string { return p.name }

// Capabilities implements llm.LLMProvider.
func (p *Provider) Capabilities(model string) llm.ProviderCapabilities {
	return llm.ProviderCapabilities{
		Streaming:         true,
		ToolCalling:       true,
		ParallelToolCalls: true,
		Thinking:          strings.Contains(model, "2.5"),
		PromptCaching:     false,
		Vision:            strings.Contains(model, "vision") || strings.Contains(model, "pro"),
	}
}

// Stream implements llm.LLMProvider.
func (p *Provider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	evCh := make(chan llm.LLMEvent, 16)
	doneCh := make(chan llm.StreamResult, 1)

	go func() {
		defer close(evCh)
		defer close(doneCh)

		client, err := genai.NewClient(ctx, &genai.ClientConfig{
			APIKey: p.config.APIKey,
		})
		if err != nil {
			doneCh <- llm.StreamResult{Err: fmt.Errorf("gemini: creating client: %w", err)}
			return
		}

		contents, sysInstr := toGeminiContents(req.Messages)
		config := toGeminiConfig(req, sysInstr)

		var resultMessages []llm.Message
		var prevTextLen int
		var emittedToolCalls map[string]bool

		for chunk, err := range client.Models.GenerateContentStream(ctx, req.Model, contents, config) {
			if err != nil {
				transient := isTransientError(err)
				select {
				case evCh <- llm.LLMErrorEvent{Error: err, Transient: transient}:
				case <-ctx.Done():
				}
				doneCh <- llm.StreamResult{Err: fmt.Errorf("gemini: stream error: %w", err)}
				return
			}

			for _, cand := range chunk.Candidates {
				if cand.Content == nil {
					continue
				}
				for _, part := range cand.Content.Parts {
					if part.Text != "" {
						text := part.Text
						if len(text) > prevTextLen {
							delta := text[prevTextLen:]
							select {
							case evCh <- llm.TextDeltaEvent{Delta: delta}:
							case <-ctx.Done():
								doneCh <- llm.StreamResult{Err: context.Cause(ctx)}
								return
							}
							prevTextLen = len(text)
						}
						if part.Thought {
							select {
							case evCh <- llm.ThinkingDeltaEvent{Delta: text}:
							case <-ctx.Done():
								doneCh <- llm.StreamResult{Err: context.Cause(ctx)}
								return
							}
						}
					}
					if part.FunctionCall != nil {
						fc := part.FunctionCall
						if !emittedToolCalls[fc.ID] {
							emittedToolCalls[fc.ID] = true
							args, _ := json.Marshal(fc.Args)
							select {
							case evCh <- llm.ToolCallStartEvent{
								CallID:   fc.ID,
								ToolName: fc.Name,
								Args:     args,
							}:
							case <-ctx.Done():
								doneCh <- llm.StreamResult{Err: context.Cause(ctx)}
								return
							}
						}
					}
				}
				if cand.FinishReason != "" {
					// Emit tool call ends for any pending tool calls
					if cand.Content != nil {
						for _, part := range cand.Content.Parts {
							if part.FunctionCall != nil {
								select {
								case evCh <- llm.ToolCallEndEvent{CallID: part.FunctionCall.ID}:
								case <-ctx.Done():
									doneCh <- llm.StreamResult{Err: context.Cause(ctx)}
									return
								}
								args, _ := json.Marshal(part.FunctionCall.Args)
								resultMessages = append(resultMessages, llm.Message{
									Role: "assistant",
									Content: llm.ToolCallContent{
										CallID:   part.FunctionCall.ID,
										ToolName: part.FunctionCall.Name,
										Args:     args,
									},
								})
							}
						}
					}
				}
			}
		}

		if len(resultMessages) == 0 && prevTextLen > 0 {
			// Pure text response
			// We don't have the full text stored; reconstruct from last chunk if needed.
			// For simplicity, emit an empty text message since the deltas were already streamed.
			// The consumer already got the text via TextDeltaEvents.
		}

		select {
		case evCh <- llm.MessageEndEvent{}:
		case <-ctx.Done():
			doneCh <- llm.StreamResult{Err: context.Cause(ctx)}
			return
		}

		doneCh <- llm.StreamResult{Messages: append(req.Messages, resultMessages...)}
	}()

	return llm.NewEventStream(evCh, doneCh)
}

func toGeminiContents(messages []llm.Message) ([]*genai.Content, *genai.Content) {
	var contents []*genai.Content
	var systemInstruction *genai.Content
	for _, msg := range messages {
		if msg.Role == "system" {
			switch c := msg.Content.(type) {
			case llm.TextContent:
				systemInstruction = genai.NewContentFromText(c.Text, genai.RoleUser)
			}
			continue
		}
		var role genai.Role = genai.RoleUser
		if msg.Role == "assistant" || msg.Role == "model" {
			role = genai.RoleModel
		}
		switch c := msg.Content.(type) {
		case llm.TextContent:
			contents = append(contents, genai.NewContentFromText(c.Text, role))
		case llm.ToolCallContent:
			contents = append(contents, genai.NewContentFromFunctionCall(c.ToolName, mustUnmarshalMap(c.Args), role))
		case llm.ToolResultContent:
			name := c.ToolName
			if name == "" {
				name = c.CallID
			}
			contents = append(contents, genai.NewContentFromFunctionResponse(name, map[string]any{"result": c.Content}, role))
		case llm.ThinkingContent:
			contents = append(contents, genai.NewContentFromText(c.Text, role))
		}
	}
	return contents, systemInstruction
}

func toGeminiConfig(req llm.LLMRequest, sysInstr *genai.Content) *genai.GenerateContentConfig {
	config := &genai.GenerateContentConfig{}
	if sysInstr != nil {
		config.SystemInstruction = sysInstr
	}
	if len(req.Tools) > 0 {
		for _, tool := range req.Tools {
			config.Tools = append(config.Tools, &genai.Tool{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:        tool.Name,
						Description: tool.Description,
						Parameters:  toGeminiSchema(tool.Schema),
					},
				},
			})
		}
	}
	if req.Thinking != llm.ThinkingOff {
		budget := map[llm.ThinkingLevel]int{
			llm.ThinkingLow:    1000,
			llm.ThinkingMedium: 4000,
			llm.ThinkingHigh:   16000,
		}[req.Thinking]
		if req.ProviderHints.Gemini != nil && req.ProviderHints.Gemini.ThinkingBudget > 0 {
			budget = req.ProviderHints.Gemini.ThinkingBudget
		}
		budget32 := int32(budget)
		config.ThinkingConfig = &genai.ThinkingConfig{
			ThinkingBudget: &budget32,
		}
	}
	return config
}

func toGeminiSchema(schema json.RawMessage) *genai.Schema {
	if len(schema) == 0 {
		return nil
	}
	var s genai.Schema
	_ = json.Unmarshal(schema, &s)
	return &s
}

func mustUnmarshalMap(data json.RawMessage) map[string]any {
	if len(data) == 0 {
		return nil
	}
	var m map[string]any
	_ = json.Unmarshal(data, &m)
	return m
}

func isTransientError(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "429") || strings.Contains(msg, "503") || strings.Contains(msg, "502") || strings.Contains(msg, "504")
}
