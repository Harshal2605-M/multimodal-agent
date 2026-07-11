# Multimodal Agent — Project Handoff Checkpoint

## 1. Purpose

This document is the authoritative continuity checkpoint for the Multimodal Agent project.

When continuing development in a new chat:

1. Use the Master Execution Plan as the source of truth for phase order, scope, and acceptance criteria.
2. Use the current repository state as the source of truth for actual implementation.
3. Use this handoff document for completed work, architectural decisions, and working style.
4. If older conversation snippets conflict with the repository or this checkpoint, prefer the current repository and this checkpoint.
5. Do not redesign completed phases without a concrete technical reason.

---

## 2. Current Checkpoint

Completed and verified:

- Phase 0 — Project setup and repository initialization.
- Phase 1 — Core contracts and data models.
- Phase 2 — FastAPI skeleton and API infrastructure.
- Phase 3 — Secure upload pipeline.
- Phase 4 — Multimodal extraction pipeline.
- Phase 5 — Deterministic preprocessing and normalization.
- Phase 6 — Resilient LLM service.

Current position:

**Start Phase 7.1 — Planner contract and prompt-security boundary.**

Do not skip ahead to LangGraph orchestration before completing Phase 7 according to the Master Execution Plan.

---

## 3. Assessment Priorities

The implementation must demonstrate the candidate's own engineering ability.

Important evaluation criteria:

- Avoid unnecessary LLM usage.
- Deterministic logic should remain deterministic.
- Tool invocation must be explicit and controlled.
- Orchestration conditions must be clear and testable.
- Follow the specified technology stack.
- Maintain production-oriented architecture.
- Optimize RAG for accuracy, relevance, and latency when the RAG phase begins.
- Handle malformed input, provider failures, unsupported inputs, malicious content, and other edge cases safely.
- Prefer readable code over unnecessary abstractions.
- Every important behavior should be explainable in an interview.

---

## 4. Frozen Working Style

Continue subphase by subphase.

For each subphase:

1. Explain what will be built.
2. Explain why it is needed and where it fits in the architecture.
3. Reuse existing contracts before creating new abstractions.
4. Implement production-quality code consistent with the repository.
5. Add focused unit/integration tests.
6. Run focused tests.
7. Run the complete test suite when the subphase affects shared behavior.
8. Explain failures in simple language.
9. Do not commit until the complete phase acceptance criteria are verified.
10. Before committing, run `git diff --check`, inspect `git status`, and ensure secrets are not staged.

The user prefers easy language and wants development to move efficiently without skipping important engineering concepts.

---

## 5. Repository Conventions

- Python 3.12.
- FastAPI application.
- Pydantic v2 models.
- `pydantic-settings` for configuration.
- `pytest` and `pytest-asyncio` for testing.
- Run tests using `python -m pytest`.
- Use dependency injection where it improves testability.
- External SDK/provider details must stay behind application-owned interfaces.
- Catch expected application/provider errors narrowly; do not hide programming bugs with broad fallback behavior.
- Validate data at application boundaries.
- Keep API keys only in `.env`.
- `.env.example` contains placeholders only.
- `.env` must remain ignored by Git.
- Use normalized application errors instead of leaking raw SDK errors.
- Use temporary upload storage with guaranteed cleanup.

---

## 6. Implemented Architecture

Current high-level flow:

User Request
→ FastAPI Boundary
→ Secure Upload Gateway
→ Extraction Orchestrator
→ Deterministic Preprocessing / NormalizedContext
→ Planner [Phase 7]
→ LangGraph Workflow [Phase 8+]
→ Tool Execution
→ Final Response

LLM boundary:

Application / Future LangGraph Nodes
→ LLMService
→ GroqProvider (primary)
→ GeminiProvider (fallback)
→ Controlled service error if both fail

---

## 7. Phase 1 — Core Contracts

Core Pydantic contracts were created and tested.

Important normalized input concepts include:

- `InputType`
- `URLType`
- `SourceMetadata`
- `ExtractedInput`
- `DetectedURL`
- `NormalizedContext`

`NormalizedContext` is the boundary passed from deterministic preprocessing into the future agent workflow.

Extracted content is untrusted data and must never automatically become system instructions.

---

## 8. Phase 2 — FastAPI Skeleton

Implemented:

- `app/main.py`
- API routes and dependencies
- `GET /`
- `GET /health`
- `POST /agent/run` placeholder
- Request ID middleware
- safe global exception handling
- basic security headers
- OpenAPI documentation

API behavior is covered by unit and integration tests.

---

## 9. Phase 3 — Secure Upload Pipeline

Implemented secure upload handling with:

- configurable file count limits
- per-file size limits
- total upload size limits
- chunked upload reads
- extension validation
- MIME/header handling
- magic-byte/file-signature validation
- renamed-file mismatch detection
- safe filenames
- temporary storage
- path traversal protection
- cleanup through the validated upload batch lifecycle
- text-only request support

Important principle:

A filename extension or client-provided Content-Type is not trusted as proof of the actual file type.

---

## 10. Phase 4 — Extraction Pipeline

Implemented multimodal extraction contracts and extraction orchestration.

Dependencies include:

- PyMuPDF
- Pillow
- pytesseract
- Tesseract OCR installed locally

Important extraction concepts:

- `ExtractionMethod`
- `ExtractedPage`
- `AudioMetadata`
- `ExtractedContent`

The extraction layer is responsible for converting validated files into normalized extracted content.

The extraction orchestrator chooses the correct extractor based on the validated detected file type.

---

## 11. Phase 5 — Deterministic Preprocessing

Implemented deterministic preprocessing/normalization before the agent planner.

Important responsibilities:

- normalize extracted content into agent-facing contracts
- preserve source identity and metadata
- deterministic URL detection
- YouTube URL classification
- build `NormalizedContext`
- keep URL detection separate from authorization to execute a tool
- preserve the security boundary between untrusted extracted content and agent instructions

Important distinction:

The extraction orchestrator decides **how to extract a validated file**.

The preprocessing layer decides **how extracted data is normalized and prepared for planning**.

---

## 12. Phase 6 — LLM Service

Implemented:

- `BaseLLMProvider`
- LLM result/provider models
- normalized LLM error hierarchy
- `GroqProvider`
- `GeminiProvider`
- `LLMService`
- plain generation
- structured generation
- Pydantic validation of structured output
- configured timeout behavior
- Groq as primary provider
- Gemini as fallback provider
- provider-used metadata
- controlled error when both providers fail
- dependency injection for unit testing

Fallback policy:

Expected normalized `LLMError`
→ fallback is allowed.

Unexpected programming error
→ propagate immediately.
→ do not hide it by calling the fallback provider.

---

## 13. Phase 6 Verification Evidence

Focused LLM tests:

`25 passed`

Complete test suite at Phase 6 checkpoint:

`237 passed, 1 warning`

The remaining warning is the Starlette TestClient/httpx deprecation warning and is not blocking Phase 6.

Real-provider verification passed:

- Real Groq plain generation.
- Real Groq structured generation.
- Real Gemini plain generation.
- Real Gemini structured generation.
- Real `LLMService` uses Groq successfully as primary.
- Forced normalized primary failure triggers real Gemini fallback.
- Controlled double failure raises `LLMServiceUnavailableError`.
- `provider_used` metadata is preserved.

Verification script:

`scripts/verify_llm_service.py`

Final verification result:

`ALL REAL PROVIDER CHECKS PASSED`

---

## 14. Current LLM Provider Behavior

`GroqProvider`:

- lazy client creation
- API key validation
- configured timeout and retries
- `generate()`
- `generate_structured()`
- JSON parsing
- Pydantic validation
- Groq timeout normalization
- generic provider error normalization
- provider-used metadata

`GeminiProvider`:

- lazy client creation
- API key validation
- configured HTTP timeout
- `generate()`
- `generate_structured()`
- Gemini response schema usage
- JSON parsing
- Pydantic validation
- known timeout handling
- generic provider error normalization
- provider-used metadata

Do not add guessed SDK-specific exception handling. Add normalization only when supported by observed SDK behavior or documentation.

---

## 15. Phase 7 — Next Phase

According to the Master Execution Plan:

**Phase 7 · Hour 14–17 — Planner + Prompt Security**

Implement:

- `app/agent/prompts.py`
- `app/agent/planner.py`

The planner receives:

- user query
- extracted content explicitly treated as untrusted data
- detected URLs
- tool definitions

The planner returns structured output containing:

- goal
- constraints
- clarification decision
- clarification question when needed
- ordered plan steps

Required engineering behavior:

- strong prompt-injection boundaries
- structured output validation
- one repair attempt when the first LLM plan is invalid
- no uncontrolled tool execution
- planner decides intended steps; execution happens later

Required scenarios include:

- PDF summarization → `summarize`
- sentiment request → `sentiment_analysis`
- code explanation → `code_explanation`
- PDF containing YouTube URL → `youtube_transcript`, then `summarize`
- Audio + PDF comparison → `compare_inputs`
- ambiguous/file-only requests → clarification
- malicious instructions inside a PDF must not override the real user query

Phase 7 commit target:

`feat: add structured agent planner`

After Phase 7:

**Phase 8 — LangGraph Skeleton + Clarification Flow**

---

## 16. Immediate Next Action

Start:

**Phase 7.1 — Define the Planner contract and prompt-security boundary.**

Before implementation:

- inspect existing `app/agent` models/contracts
- inspect `NormalizedContext`
- inspect current LLM structured-generation contract
- reuse existing models where possible
- do not duplicate contracts
- define exactly what the planner may decide
- define exactly what the planner may not execute
- define how untrusted extracted content is represented in prompts
- define the single repair-attempt policy

Then implement Phase 7 subphase by subphase according to the Master Execution Plan.