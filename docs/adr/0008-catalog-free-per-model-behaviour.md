# ADR-0008 — Per-model behaviour without a model catalog

**Status:** Accepted (2026-06-26)
**Context repos:** `pi-llm-go`

## Context

Upstream `packages/ai` derives all per-model behaviour from a generated catalog
(`models.generated.ts`): each entry carries `reasoning` (does it think), input modalities
(vision), context window, cost, and a `compat` object selecting the wire-level thinking
mechanism (`thinkingFormat`) and quirks (`maxTokensField`,
`requiresReasoningContentOnAssistantMessages`). The provider code receives a `Model`
object, not a bare id, and branches on catalog metadata.

ADR-0001 already declined the upstream 30-provider matrix; by extension we ship no model
catalog. Our providers receive a bare model **string**. Two needs surfaced by the
0.79.1→0.79.10 re-diff forced the question of where per-model behaviour comes from without
a catalog:

1. **Gemini capability + thinking-config correctness.** Gemini 2.5 drives reasoning via
   `thinkingBudget` (tokens) while Gemini 3.x / Gemma 4 use `thinkingLevel` (enum). The
   prior substring heuristic (`Contains(model, "2.5")`) mis-reported both `Thinking`/`Vision`
   capabilities and sent the wrong thinking-config field for the Gemini 3 generation the
   user actually runs (`gemini-3.1-pro`, `gemini-3.5-flash`).
2. **OpenAI-compatible thinking formats.** Running DeepSeek V4 on the `opencode-go`
   gateway (`api: openai-completions`, `baseUrl: https://opencode.ai/zen/go/v1`) requires
   the `deepseek` `thinkingFormat` (`thinking:{type}` + `reasoning_effort`) plus a plain
   `reasoning_content` round-trip — none of which the `reasoning_effort`-only path covers.

Upstream `main` also carries an unshipped `[Unreleased]` overhaul (`createModels()` /
`Models` runtime, `ProviderAuth`) moving toward a first-class model registry — i.e.
upstream is investing **further** in the catalog, the opposite direction from us.

## Decision

Per-model behaviour is derived **without a catalog**, by two catalog-free mechanisms:

- **Gemini — generation classifier.** A `classifyGemini(model string)` helper maps the
  model string to a generation (`legacy` / `2.5` / `gemini-3-pro` / `gemini-3-flash` /
  `gemma-4`), mirroring upstream's `isGemini3ProModel` / `isGemini3FlashModel` /
  `isGemma4Model` regexes. The classification drives both `ProviderCapabilities`
  (`Thinking`, `Vision`) and the thinking config (`thinkingLevel` enum for Gemini 3 /
  Gemma 4, `thinkingBudget` tokens for 2.5, `IncludeThoughts` when on, lowest-level-without-
  thoughts for off on Gemini 3).
- **OpenAI-compatible — caller-supplied `Compat`.** A `Compat` config on
  `OpenAICompatConfig` mirrors upstream's `compat` object: `ThinkingFormat` (enum),
  `SupportsReasoningEffort`, `RequiresReasoningContentOnAssistantMessages`, `MaxTokensField`.
  Because there is no catalog to populate it from, the **caller** sets the `Compat` matching
  their model when it deviates from the OpenAI default.

We do **not** adopt the upstream `Models` runtime / model catalog.

## Alternatives considered

1. **Port the upstream catalog (`models.generated.ts`).** Rejected: a multi-thousand-line
   generated artifact spanning 30+ providers, regenerated from upstream's build. Carrying it
   re-introduces exactly the breadth ADR-0001 rejected, plus a regeneration pipeline we'd
   own. Most of its value (pricing, context windows for providers we don't ship) is dead
   weight.
2. **Hardcode each model in provider `switch` statements.** Rejected on the OpenAI-compat
   side: a hardcoded `deepseek` branch repeats for every next format (chat-template, zai,
   qwen). The `Compat` seam absorbs new formats as one enum value + one branch, configured by
   the caller who knows their deployment.
3. **Wait for the upstream `Models` runtime (`[Unreleased]`) and adopt it.** Rejected for
   now: unshipped; adopting an unreleased API violates the milestone re-diff policy. Recorded
   as a watch-item below.

## Consequences

- New Gemini generations (Gemini 4, …) need a one-line classifier edit, caught at the next
  milestone re-diff rather than silently wrong forever. The allowlist's bounded maintenance
  cost was preferred over a denylist's silent-misclassification risk.
- New OpenAI-compatible thinking formats are a `ThinkingFormat` enum value + one branch; the
  caller opts in per model via `Compat`. No catalog, no per-model metadata table.
- The caller owns per-model correctness on the OpenAI-compat side — they must set `Compat` to
  match their model. Documented on `OpenAICompatConfig`; without it, a model gets OpenAI-
  default behaviour.
- Divergence from upstream is deliberate and will **widen** when the `[Unreleased]`
  `Models`/`ProviderAuth` overhaul ships. **WATCH-ITEM:** the next milestone re-diff must
  begin by checking whether that overhaul has released and re-evaluating this ADR against it —
  at minimum whether `ProviderAuth` offers anything `GetAPIKey` does not, and whether the
  `Models` runtime changes the cost/benefit of staying catalog-free.
- When diffing the Go port against upstream, the absence of `models.generated.ts` and the
  `compat`-object catalog is intentional; this ADR is the record of why.
