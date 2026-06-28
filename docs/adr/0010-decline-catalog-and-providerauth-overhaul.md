# ADR-0010 — Decline the upstream catalog / `ProviderAuth` overhaul

**Status:** Accepted (2026-06-26)
**Context repos:** `resolute-llm-go`, `resolute-agent-core-go`
**Relates to:** re-affirms ADR-0008 (catalog-free per-model behaviour); preserves ADR-0001
(two `LLMProvider` implementations) and ADR-0009 (Vertex backend). Closes the watch-item recorded
in `REDIFF-0.79.1-to-0.79.10.md` and `project_07910_rediff`.

## Context

`REDIFF-0.79.1-to-0.79.10.md` flagged an upstream `[Unreleased]` overhaul —
`Models`/`ProviderAuth`/`createModels()` — as the watch-item for the next milestone, because it
moves upstream toward a first-class model catalog, the opposite of ADR-0008. That overhaul
**shipped in upstream v0.80.0** (2026-06-23) and is the bulk of the 0.79.10→0.80.2 delta
(`REDIFF-0.79.10-to-0.80.2.md`). It comprises:

- A **first-class model catalog**: a `.models.ts` per ~30 providers, plus `createModels()` /
  `builtinModels()` runtime, sync model reads, async `refresh()`, and `Provider` renamed to
  `ProviderId` with `Provider` now naming a runtime provider interface.
- A **`ProviderAuth` substrate**: `CredentialStore` (serialized read/modify/delete), `envApiKeyAuth()`,
  `lazyOAuth()`, injectable `AuthContext`, and OAuth refresh under a store lock.
- A **breaking change in `packages/agent`**: `AgentHarnessOptions.models` is required and the only
  auth path; `getApiKeyAndHeaders` is removed; `compact()`/`generateSummary()`/`generateBranchSummary()`
  take a `Models` parameter instead of `apiKey`/`headers`.

This machinery exists to solve TypeScript-ecosystem problems: tree-shaking, side-effect-free
entrypoints, isolated bundles, and a 30-provider matrix. The Go port has **two** provider
implementations (ADR-0001) and none of those problems.

The agent breaking change, examined against the Go port, is already satisfied: auth lives on each
`llm.LLMProvider` (`Config.APIKey`/`GetAPIKey`), the agent holds a provider registry
(`config.Providers`, `providerByName`), and `compact.go`'s `summarize*` functions already take a
`provider llm.LLMProvider` and stream through it. The port was already at the destination upstream
is migrating toward.

## Decision

**Decline the catalog and `ProviderAuth` overhaul.** Stay catalog-free (ADR-0008): per-model
behaviour (thinking, vision, tool-calling, context handling) is derived at runtime by a classifier,
not read from a static catalog. Keep auth owned by each provider via `Config.APIKey`/`GetAPIKey`
plus the Vertex ADC toggle (ADR-0009). Do not adopt `createModels()`, `CredentialStore`, OAuth
refresh, or the `/api/*` restructure.

This **closes the catalog watch-item.** ADR-0001's two-implementation decision and ADR-0008's
catalog-free decision both stand, now explicitly re-affirmed against the shipped upstream catalog.

Adding more providers (xAI, Mistral, Qwen, z.ai) does **not** require this overhaul: they are
configured instances of the existing `openai-compat` adapter (ADR-0001 already scopes it to "any
OpenAI-compatible endpoint"), each classified at runtime per ADR-0008.

## Alternatives considered

1. **Adopt the overhaul wholesale.** Rejected: it reverses ADR-0008, re-litigates ADR-0009's
   Vertex auth, and rewrites both repos for **zero behavioural gain** on the two in-scope providers.
   The benefits it delivers (bundle tree-shaking, a 30-provider catalog, OAuth credential stores)
   address problems the Go port does not have (CLAUDE.md rules 2 and 5).
2. **Adopt only the `ProviderAuth` substrate, skip the catalog.** Rejected: the substrate's value is
   OAuth credential management; every in-scope provider authenticates with a static API key or
   Vertex ADC, both already covered. A `CredentialStore` with one static credential per provider is
   indirection without a payoff (deletion test: removing it makes complexity vanish, not reappear).
3. **Adopt only the agent breaking change for upstream-shape parity.** Rejected: it is a no-op —
   the Go agent is already provider-owned and already streams compaction through the provider.
   Renaming to match upstream's TS surface buys nothing and churns a stable API.

## Consequences

- **Catalog-free holds; the watch-item is closed.** The next re-diff starts from a settled position
  rather than re-opening this each milestone.
- **One thing is forgone: OAuth provider auth** (token refresh, credential store, per-provider typed
  credentials). No provider in the current set (Gemini, OpenAI, opencode-go, xAI, Mistral, Qwen,
  z.ai) needs it — all use a static key or ADC.
- **New watch-item (narrow):** if a future *in-scope* provider requires OAuth (e.g.
  Anthropic-via-OAuth, GitHub Copilot), revisit the `ProviderAuth` decline at that point. This is the
  only condition that reopens this ADR.
- **No agent-side change.** `resolute-agent-core-go` ports nothing from the 0.80.0 agent breaking
  change; its existing `LLMProvider`-registry shape is unaffected.
- **Mirrors** into `resolute-llm-go/docs/adr/` per the 0001/0002/0005/0008 convention once the repo
  rename (W1) lands.
