// Package gemini provides an LLMProvider implementation built on the official
// Google GenAI SDK (google.golang.org/genai).
package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"

	"github.com/dev-resolute/resolute-llm-go"
	"google.golang.org/genai"
)

// Config holds the Gemini provider configuration.
type Config struct {
	APIKey    string
	GetAPIKey func(ctx context.Context) (string, error)
	Retry     llm.RetryPolicy

	// Vertex routes requests through the Vertex AI backend instead of the Gemini
	// API. Vertex authenticates with Application Default Credentials (e.g. GKE
	// Workload Identity), so no API key is used — APIKey and GetAPIKey are ignored
	// when Vertex is set.
	Vertex bool
	// Project is the GCP project ID for the Vertex AI backend. Used only when
	// Vertex is set; falls back to the GOOGLE_CLOUD_PROJECT environment variable.
	Project string
	// Location is the GCP region for the Vertex AI backend (e.g. "us-central1" or
	// "global"). Used only when Vertex is set; falls back to the
	// GOOGLE_CLOUD_LOCATION environment variable.
	Location string
}

// clientConfig builds the genai client config for the chosen backend. The Vertex
// backend uses Application Default Credentials, so it carries no API key.
func (cfg Config) clientConfig(apiKey string) *genai.ClientConfig {
	if cfg.Vertex {
		return &genai.ClientConfig{
			Backend:  genai.BackendVertexAI,
			Project:  cfg.Project,
			Location: cfg.Location,
		}
	}
	return &genai.ClientConfig{APIKey: apiKey}
}

// Provider implements llm.LLMProvider for Google Gemini models.
type Provider struct {
	name     string
	config   Config
	client   *genai.Client
	lastKey  string
	clientMu sync.Mutex
}

// New creates a Provider from the given Config. With Config.Vertex it targets
// the Vertex AI backend (ADC credentials); otherwise the Gemini API with APIKey.
func New(cfg Config) (llm.LLMProvider, error) {
	client, err := genai.NewClient(context.Background(), cfg.clientConfig(cfg.APIKey))
	if err != nil {
		return nil, fmt.Errorf("gemini: creating client: %w", err)
	}
	return &Provider{name: "gemini", config: cfg, client: client}, nil
}

// Name implements llm.LLMProvider.
func (p *Provider) Name() string { return p.name }

// Capabilities implements llm.LLMProvider.
func (p *Provider) Capabilities(model string) llm.ProviderCapabilities {
	class := classifyGemini(model)
	return llm.ProviderCapabilities{
		Streaming:         true,
		ToolCalling:       true,
		ParallelToolCalls: true,
		Thinking:          class.thinks(),
		PromptCaching:     false,
		Vision:            class.vision(),
	}
}

// geminiClass is the generation of a Gemini model, derived from its id without a
// model catalog (ADR-0008). It selects both the reported capabilities and the
// thinking-config mechanism.
type geminiClass int

const (
	classLegacy geminiClass = iota // pre-2.5: no thinking
	class25                        // Gemini 2.5: thinkingBudget tokens
	class3Pro                      // Gemini 3.x pro: thinkingLevel enum
	class3Flash                    // Gemini 3.x flash + flash-latest aliases: thinkingLevel enum
	classGemma4                    // Gemma 4: thinkingLevel enum
)

var (
	reGemini3Pro   = regexp.MustCompile(`gemini-3(?:\.\d+)?-pro`)
	reGemini3Flash = regexp.MustCompile(`gemini-3(?:\.\d+)?-flash`)
	reGemma4       = regexp.MustCompile(`gemma-?4`)
)

// classifyGemini maps a model id to its generation, mirroring upstream google.ts's
// isGemini3ProModel/isGemini3FlashModel/isGemma4Model predicates.
func classifyGemini(model string) geminiClass {
	m := strings.ToLower(model)
	switch {
	case reGemma4.MatchString(m):
		return classGemma4
	case reGemini3Pro.MatchString(m):
		return class3Pro
	case reGemini3Flash.MatchString(m) || m == "gemini-flash-latest" || m == "gemini-flash-lite-latest":
		return class3Flash
	case strings.Contains(m, "2.5"):
		return class25
	default:
		return classLegacy
	}
}

func (c geminiClass) thinks() bool { return c != classLegacy }

func (c geminiClass) vision() bool {
	return c == class25 || c == class3Pro || c == class3Flash
}

func (c geminiClass) usesThinkingLevel() bool {
	return c == class3Pro || c == class3Flash || c == classGemma4
}

// strictToolSamplingSupported reports whether the model enforces required
// function parameters in validated tool-calling mode (Gemini 3+; upstream
// google-shared.ts supportsGoogleStrictToolSampling — Gemma is excluded,
// its ids don't match upstream's ^gemini- gate).
func strictToolSamplingSupported(model string) bool {
	switch classifyGemini(model) {
	case class3Pro, class3Flash:
		return true
	}
	return false
}

// disabledThinkingLevel is the lowest level used when thinking is "off" for models
// that cannot fully disable it (Gemini 3): pro clamps to LOW, flash/Gemma to MINIMAL.
func (c geminiClass) disabledThinkingLevel() genai.ThinkingLevel {
	if c == class3Pro {
		return genai.ThinkingLevelLow
	}
	return genai.ThinkingLevelMinimal
}

func thinkingLevelFor(level llm.ThinkingLevel) genai.ThinkingLevel {
	switch level {
	case llm.ThinkingLow:
		return genai.ThinkingLevelLow
	case llm.ThinkingMedium:
		return genai.ThinkingLevelMedium
	case llm.ThinkingHigh, llm.ThinkingXhigh, llm.ThinkingMax:
		return genai.ThinkingLevelHigh
	default:
		return genai.ThinkingLevelMinimal
	}
}

// thinkingConfigFor builds the thinking config for a thinking-on request: a
// thinkingLevel enum for Gemini 3 / Gemma 4, thinkingBudget tokens for Gemini 2.5,
// and nil for models that do not reason.
func thinkingConfigFor(req llm.LLMRequest) *genai.ThinkingConfig {
	class := classifyGemini(req.Model)
	if !class.thinks() {
		return nil
	}
	if class.usesThinkingLevel() {
		return &genai.ThinkingConfig{
			ThinkingLevel:   thinkingLevelFor(req.Thinking),
			IncludeThoughts: true,
		}
	}
	budget := map[llm.ThinkingLevel]int{
		llm.ThinkingMinimal: 512,
		llm.ThinkingLow:     1000,
		llm.ThinkingMedium:  4000,
		llm.ThinkingHigh:    16000,
		llm.ThinkingXhigh:   16000,
		llm.ThinkingMax:     16000,
	}[req.Thinking]
	if b, ok := req.ThinkingBudgets[req.Thinking]; ok {
		budget = b
	}
	if req.ProviderHints.Gemini != nil && req.ProviderHints.Gemini.ThinkingBudget > 0 {
		budget = req.ProviderHints.Gemini.ThinkingBudget
	}
	budget32 := int32(budget)
	return &genai.ThinkingConfig{
		ThinkingBudget:  &budget32,
		IncludeThoughts: true,
	}
}

// Stream implements llm.LLMProvider.
func (p *Provider) Stream(ctx context.Context, req llm.LLMRequest) llm.EventStream {
	return llm.Run(ctx, req, p.produce)
}

func (p *Provider) produce(ctx context.Context, req llm.LLMRequest, emit func(llm.LLMEvent) error, headers map[string]string, setResponseMeta func(status int, respHeaders map[string]string)) ([]llm.Message, error) {
	if req.Transport == llm.TransportWebSocket {
		return nil, fmt.Errorf("gemini: %w: %s", llm.ErrTransportUnsupported, req.Transport)
	}

	apiKey := p.config.APIKey
	if p.config.GetAPIKey != nil {
		key, err := p.config.GetAPIKey(ctx)
		if err != nil {
			return nil, fmt.Errorf("gemini: refreshing api key: %w", err)
		}
		if key != "" {
			apiKey = key
		}
	}

	// Rebuild client if the resolved key changed from what the pooled client
	// was constructed with. This is the fallback path documented in the PRD.
	if apiKey != "" && apiKey != p.lastKey {
		p.clientMu.Lock()
		if apiKey != p.lastKey {
			client, err := genai.NewClient(ctx, &genai.ClientConfig{
				APIKey: apiKey,
			})
			if err != nil {
				p.clientMu.Unlock()
				return nil, fmt.Errorf("gemini: rebuilding client for new key: %w", err)
			}
			p.client = client
			p.lastKey = apiKey
		}
		p.clientMu.Unlock()
	}

	// TODO(v0.x): per-call header injection via genai request options if the SDK
	// exposes them; otherwise headers are a no-op for Gemini.
	_ = headers

	contents, sysInstr := toGeminiContents(req.Messages, req.Model)
	config, err := toGeminiConfig(req, sysInstr)
	if err != nil {
		if emitErr := emit(llm.LLMErrorEvent{Error: err, Transient: false}); emitErr != nil {
			return nil, emitErr
		}
		return nil, err
	}

	var resultMessages []llm.Message
	var pendingToolCalls []llm.ToolCallContent
	emittedToolCalls := map[string]bool{}
	var finishReason genai.FinishReason

	// flushToolCalls ends every accumulated tool call and moves it into the
	// result messages. Tool-call parts and the finish reason can arrive in
	// different chunks, so calls are collected as parts stream in and flushed
	// on finish (or at stream end as a fallback).
	flushToolCalls := func() error {
		for _, call := range pendingToolCalls {
			if err := emit(llm.ToolCallEndEvent{
				CallID:           call.CallID,
				ToolName:         call.ToolName,
				Args:             call.Args,
				ThoughtSignature: call.ThoughtSignature,
			}); err != nil {
				return err
			}
			resultMessages = append(resultMessages, llm.Message{Role: "assistant", Content: call})
		}
		pendingToolCalls = nil
		return nil
	}

	for chunk, err := range p.client.Models.GenerateContentStream(ctx, req.Model, contents, config) {
		if err != nil {
			err = classifyStreamError(err)
			transient := isTransientError(err)
			if emitErr := emit(llm.LLMErrorEvent{Error: err, Transient: transient}); emitErr != nil {
				return nil, emitErr
			}
			return nil, fmt.Errorf("gemini: stream error: %w", err)
		}

		for _, cand := range chunk.Candidates {
			if cand.Content == nil {
				continue
			}
			for _, part := range cand.Content.Parts {
				if part.Text != "" {
					if part.Thought {
						if err := emit(llm.ThinkingDeltaEvent{Delta: part.Text}); err != nil {
							return nil, err
						}
					} else if err := emit(llm.TextDeltaEvent{Delta: part.Text}); err != nil {
						return nil, err
					}
				}
				if part.FunctionCall != nil {
					fc := part.FunctionCall
					if !emittedToolCalls[fc.ID] {
						emittedToolCalls[fc.ID] = true
						args, _ := json.Marshal(fc.Args)
						call := llm.ToolCallContent{
							CallID:           fc.ID,
							ToolName:         fc.Name,
							Args:             args,
							ThoughtSignature: part.ThoughtSignature,
						}
						if err := emit(llm.ToolCallStartEvent{
							CallID:           call.CallID,
							ToolName:         call.ToolName,
							Args:             call.Args,
							ThoughtSignature: call.ThoughtSignature,
						}); err != nil {
							return nil, err
						}
						pendingToolCalls = append(pendingToolCalls, call)
					}
				}
			}
			if cand.FinishReason != "" {
				finishReason = cand.FinishReason
				if err := flushToolCalls(); err != nil {
					return nil, err
				}
			}
		}
	}

	// Fallback for streams that end without a finish-reason chunk.
	if err := flushToolCalls(); err != nil {
		return nil, err
	}

	if err := emit(llm.MessageEndEvent{StopReason: mapGeminiFinishReason(finishReason, len(emittedToolCalls) > 0)}); err != nil {
		return nil, err
	}

	return resultMessages, nil
}

// mapGeminiFinishReason maps a Gemini candidate finish reason to the portable
// StopReason. Length wins over tool use: a MAX_TOKENS-truncated message's
// calls may be incomplete (upstream #6285).
func mapGeminiFinishReason(reason genai.FinishReason, sawToolCalls bool) llm.StopReason {
	switch {
	case reason == genai.FinishReasonMaxTokens:
		return llm.StopReasonLength
	case sawToolCalls:
		return llm.StopReasonToolUse
	case reason == genai.FinishReasonStop:
		return llm.StopReasonStop
	default:
		return llm.StopReasonUnknown
	}
}

func toGeminiContents(messages []llm.Message, model string) ([]*genai.Content, *genai.Content) {
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
			part := genai.NewPartFromFunctionCall(c.ToolName, mustUnmarshalMap(c.Args))
			part.ThoughtSignature = c.ThoughtSignature
			contents = append(contents, &genai.Content{Role: string(role), Parts: []*genai.Part{part}})
		case llm.ToolResultContent:
			name := c.ToolName
			if name == "" {
				name = c.CallID
			}
			result := c.Content
			if result == "" && len(c.Images) > 0 {
				result = "(see attached image)"
			}
			// Use "output" for success and "error" for errors, per the SDK
			// documentation and upstream google-shared.ts.
			response := map[string]any{"output": result}
			if c.IsError {
				response = map[string]any{"error": result}
			}
			fr := &genai.FunctionResponse{Name: name, Response: response}
			nestImages := supportsMultimodalFunctionResponse(model)
			if len(c.Images) > 0 && nestImages {
				for _, img := range c.Images {
					fr.Parts = append(fr.Parts, &genai.FunctionResponsePart{
						InlineData: &genai.FunctionResponseBlob{MIMEType: img.MimeType, Data: img.Data},
					})
				}
			}
			frPart := &genai.Part{FunctionResponse: fr}
			// Consecutive tool results share one user turn (upstream merges so
			// Cloud Code Assist-style backends see a single functionResponse turn).
			if last := lastContent(contents); last != nil && last.Role == "user" && hasFunctionResponse(last) {
				last.Parts = append(last.Parts, frPart)
			} else {
				contents = append(contents, &genai.Content{Role: "user", Parts: []*genai.Part{frPart}})
			}
			// Pre-Gemini-3 models don't support multimodal function responses:
			// images go in a separate user turn (which ends the merge run, as
			// upstream's does).
			if len(c.Images) > 0 && !nestImages {
				parts := []*genai.Part{{Text: "Tool result image:"}}
				for _, img := range c.Images {
					parts = append(parts, &genai.Part{InlineData: &genai.Blob{MIMEType: img.MimeType, Data: img.Data}})
				}
				contents = append(contents, &genai.Content{Role: "user", Parts: parts})
			}
		case llm.ImageContent:
			contents = append(contents, &genai.Content{
				Role:  string(role),
				Parts: []*genai.Part{{InlineData: &genai.Blob{MIMEType: c.MimeType, Data: c.Data}}},
			})
		case llm.ThinkingContent:
			contents = append(contents, genai.NewContentFromText(c.Text, role))
		}
	}
	return contents, systemInstruction
}

// supportsMultimodalFunctionResponse reports whether the model accepts images
// nested inside functionResponse.parts (Gemini 3+ / Gemma 4; upstream
// google-shared.ts gates on Gemini major version >= 3).
func supportsMultimodalFunctionResponse(model string) bool {
	switch classifyGemini(model) {
	case class25, classLegacy:
		return false
	}
	return true
}

func lastContent(contents []*genai.Content) *genai.Content {
	if len(contents) == 0 {
		return nil
	}
	return contents[len(contents)-1]
}

func hasFunctionResponse(c *genai.Content) bool {
	for _, p := range c.Parts {
		if p.FunctionResponse != nil {
			return true
		}
	}
	return false
}

// toGeminiConfig builds the genai request config for req. An error return
// means a tool requested "require" strict sampling on a model that doesn't
// support validated function calling, or carries an invalid Strict value
// (llm.ResolveStrictSampling); the error is pre-wrapped in llm.ErrProviderFatal
// (LLM-11 shape) since it is deterministic and will fail identically on retry —
// the caller must treat it as a fatal pre-flight error.
func toGeminiConfig(req llm.LLMRequest, sysInstr *genai.Content) (*genai.GenerateContentConfig, error) {
	config := &genai.GenerateContentConfig{}
	if sysInstr != nil {
		config.SystemInstruction = sysInstr
	}
	if len(req.Tools) > 0 {
		supported := strictToolSamplingSupported(req.Model)
		var anyStrict bool
		for _, tool := range req.Tools {
			strict, err := llm.ResolveStrictSampling(tool, supported)
			if err != nil {
				return nil, fmt.Errorf("%w: %w", llm.ErrProviderFatal, err)
			}
			if strict {
				anyStrict = true
			}
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
		// VALIDATED is request-level (upstream google-shared.ts:311-324), not
		// per-tool: any tool resolving strict turns it on for the whole call.
		// Our adapter exposes no toolChoice, so upstream's toolChoice interplay
		// is N/A here.
		if anyStrict {
			config.ToolConfig = &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeValidated,
				},
			}
		}
	}
	if req.Thinking != llm.ThinkingOff {
		config.ThinkingConfig = thinkingConfigFor(req)
	} else if class := classifyGemini(req.Model); class.usesThinkingLevel() {
		config.ThinkingConfig = &genai.ThinkingConfig{
			ThinkingLevel:   class.disabledThinkingLevel(),
			IncludeThoughts: false,
		}
	}
	return config, nil
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

// classifyStreamError wraps deterministic client errors in llm.ErrProviderFatal
// so retry ladders stop retrying requests Gemini will reject identically every
// time (LLM-11): HTTP 400/401/403/404 and the INVALID_ARGUMENT /
// FAILED_PRECONDITION statuses. Quota (429), server (5xx), and transport errors
// pass through unchanged, as do context-overflow 400s — callers handle those by
// compacting and retrying (LLM-8), which a fatal wrap could suppress.
func classifyStreamError(err error) error {
	if err == nil {
		return nil
	}
	var apiErr genai.APIError
	if !errors.As(err, &apiErr) {
		return err
	}
	if errors.Is(llm.AsContextOverflow(err), llm.ErrContextOverflow) {
		return err
	}
	switch apiErr.Code {
	case 400, 401, 403, 404:
		return fmt.Errorf("%w: %w", llm.ErrProviderFatal, err)
	}
	switch apiErr.Status {
	case "INVALID_ARGUMENT", "FAILED_PRECONDITION":
		return fmt.Errorf("%w: %w", llm.ErrProviderFatal, err)
	}
	return err
}
