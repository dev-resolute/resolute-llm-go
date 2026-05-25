# pi-llm-go

Provider-agnostic LLM abstraction for Go, extracted from the Pi agent framework.

## Features

- **Single interface**: `LLMProvider` abstracts OpenAI-compatible and Gemini endpoints.
- **Normalized streaming**: `EventStream` with typed `LLMEvent` variants.
- **Portable thinking**: `ThinkingLevel` enum with per-provider mapping.
- **First-class mock**: `mock.MockProvider` with fluent builder for tests.
- **No agent opinions**: Just the LLM call layer; bring your own loop.

## Providers

- `openai-compat` — OpenAI, Fireworks, Ollama, vLLM, llama.cpp server, LM Studio
- `gemini` — Google Gemini via `google.golang.org/genai`

## Install

```bash
go get github.com/resolute-sh/pi-llm-go
```

## Usage

```go
import (
    "github.com/resolute-sh/pi-llm-go"
    "github.com/resolute-sh/pi-llm-go/openai-compat"
)

provider, _ := openaicompat.New(openaicompat.Config{
    BaseURL: "https://api.openai.com/v1",
    APIKey:  os.Getenv("OPENAI_API_KEY"),
})

stream := provider.Stream(ctx, llm.LLMRequest{
    Model:    "gpt-4o",
    Messages: []llm.Message{{Role: "user", Content: llm.TextContent{Text: "Hello"}}},
})

for ev := range stream.Events {
    // type-switch on llm.LLMEvent
}
result := <-stream.Done
```

## Testing

Unit tests pass without secrets:
```bash
go test ./...
```

Integration tests (require API keys):
```bash
go test -tags=integration ./...
```

## License

MIT
