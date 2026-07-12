# Multimodal AI Agent Architecture

## 1. Architecture Overview

The Multimodal AI Agent is a constrained, tool-using AI system that accepts
text queries, uploaded multimodal inputs, and validated YouTube URLs.

The system separates deterministic preprocessing, LLM-based planning,
semantic plan validation, controlled tool execution, and response composition.

The architecture is designed around explicit trust boundaries:

- User queries are trusted only as task requests.
- Uploaded and extracted content is treated as untrusted data.
- URLs are detected and validated before planner execution.
- The planner can select only application-owned allowlisted tools.
- Plan steps are structurally and semantically validated before execution.
- The executor resolves structured input references instead of allowing
  arbitrary model-generated arguments.
- Tools return standardized results.
- Execution traces exclude raw extracted content and provider exceptions.

## 2. High-Level Request Flow

```text
Client / Browser
        |
        v
FastAPI API Boundary
        |
        v
Request Validation and Secure Upload Processing
        |
        v
Multimodal Preprocessing
        |
        +---- PDF ------> PyMuPDF -----> OCR fallback when required
        |
        +---- Image ----> Tesseract OCR
        |
        +---- Audio ----> faster-whisper
        |
        +---- URL ------> URL Detection and Validation
        |
        v
NormalizedContext
        |
        v
Planner
        |
        +---- Groq Primary Provider
        |
        +---- Gemini Fallback Provider
        |
        v
PlannerOutput Schema Validation
        |
        v
Semantic Plan Validation
        |
        +---- Invalid Plan ---> Bounded Repair Attempt
        |
        v
LangGraph Workflow
        |
        v
Executor
        |
        v
Input Reference Resolution
        |
        v
Tool Registry
        |
        +---- summarize
        +---- sentiment_analysis
        +---- code_explanation
        +---- youtube_transcript
        +---- compare_inputs
        +---- conversational_answer
        |
        v
ToolResult
        |
        v
Response Composer
        |
        v
Structured API Response + Safe Trace
        |
        v
Frontend UI
3. Core Components
FastAPI API Boundary

FastAPI provides the HTTP boundary for the application.

The API layer is responsible for request validation, request identifiers,
upload handling, application dependency access, workflow invocation, and
structured responses.

The frontend is served as static HTML, CSS, and JavaScript by the same
application.

Multimodal Preprocessing

Preprocessing converts heterogeneous inputs into a normalized representation
before planner execution.

Supported preprocessing paths include:

PDF text extraction using PyMuPDF.
OCR fallback for image-based PDF content when required.
Image OCR using Tesseract.
Audio transcription using faster-whisper.
YouTube URL detection and validation.

Extracted content is stored in NormalizedContext and remains separated by
source identifiers.

NormalizedContext

NormalizedContext is the boundary between preprocessing and agent reasoning.

It contains:

The user's query.
Extracted inputs with source identifiers.
Validated detected URLs.
Request-level normalized data required by downstream components.

The planner reasons over this normalized representation instead of raw HTTP
uploads.

Planner

The planner converts the user request and normalized context into a constrained
PlannerOutput.

The planner receives:

Trusted application policy.
The application-owned tool allowlist.
The user's request.
Validated URL metadata.
Explicitly delimited untrusted extracted content.

The planner does not execute tools.

LLM Provider Strategy

Groq is the primary LLM provider.

Gemini is used as the fallback provider when the primary provider is
unavailable or generation fails.

Provider-specific implementation details are hidden behind the application LLM
provider contract.

Structured outputs are validated against application-owned Pydantic models
before they are accepted by the workflow.

Plan Validation and Repair

Planner output passes through two validation layers.

Pydantic schema validation checks structural correctness.
Semantic plan validation checks the plan against the current
NormalizedContext and complete execution plan.

Semantic validation checks include:

Maximum plan-step limits.
Unique step identifiers.
Existing source references.
Valid source collections.
Existing detected URLs.
Valid previous-step references.
Dependency ordering.
Dependency declarations.

A bounded repair attempt allows a schema-valid but semantically invalid plan
to be corrected without permitting unbounded agent loops.

LangGraph Workflow

LangGraph coordinates explicit workflow states and transitions.

The workflow includes:

Planner node.
Plan-validation node.
Clarification node.
Executor node.
Response-composer node.

Routing decisions are application-owned and deterministic.

LangGraph is used for workflow orchestration rather than unrestricted agent
autonomy.

Executor and Input Resolution

The executor processes one validated plan step at a time.

The model does not generate arbitrary tool arguments.

Instead, each plan step contains a structured InputReference, such as:

source
sources
all_sources
step_output
detected_urls
query_context

The executor deterministically resolves these references against the current
runtime state and creates an application-owned ToolInput.

Tool Registry

The Tool Registry is the authoritative runtime mapping between allowlisted
ToolName values and tool implementations.

The six agent-callable tools are:

summarize
sentiment_analysis
code_explanation
youtube_transcript
compare_inputs
conversational_answer

Unknown or arbitrary tool names cannot be executed.

Response Composer

The response composer converts completed workflow state into the final
user-facing response.

It uses successful tool outputs, controlled failures, clarification state, and
safe execution metadata.

The frontend receives a structured response containing the final answer and
safe observability information.

4. Security Design
Prompt-Injection Boundary

Uploaded files, OCR text, PDF text, audio transcripts, and other extracted
content are treated as untrusted data.

Instructions found inside extracted content cannot override planner policy,
modify the user request, add tools, or change tool behavior.

Tool Allowlisting

Only values defined by the application-owned ToolName enum and registered in
the Tool Registry can execute.

The planner cannot request shell execution, arbitrary code execution, generic
web search, or unrestricted URL fetching.

Structured Input References

Planner steps reference application-owned context through constrained
InputReference values.

This prevents the model from directly constructing arbitrary runtime tool
arguments.

Upload Safety

Uploads are validated against application limits and supported input types.

Temporary processing files are cleaned after request processing and are not
committed to source control.

Secret Management

LLM API keys are provided through environment variables.

Secrets are excluded from Git, Docker images, application traces, and
user-facing responses.

Safe Tracing

Execution traces expose workflow stages, selected tools, statuses, step IDs,
and controlled error codes.

Raw extracted content, API keys, model responses, and provider exceptions are
not included in user-facing traces.

5. Deployment Architecture
GitHub Repository
        |
        v
Render Docker Build
        |
        v
Docker Image
  - Python runtime
  - Application dependencies
  - Tesseract
  - FFmpeg
        |
        v
Render Web Service
        |
        +---- Environment Secrets
        |
        +---- Public HTTPS Endpoint
        |
        +---- /health
        |
        v
Browser UI / API Client

The same Docker image is verified locally before public deployment.

Render provides the runtime PORT, while the application binds Uvicorn to
0.0.0.0.

6. Key Design Decisions
Use deterministic preprocessing before LLM planning.
Treat extracted multimodal content as untrusted data.
Use LangGraph for explicit stateful orchestration.
Use a constrained planner instead of unrestricted agent autonomy.
Validate plans structurally and semantically.
Use structured input references instead of model-generated tool arguments.
Execute only allowlisted tools from an application-owned registry.
Use Groq as the primary provider with Gemini fallback.
Keep provider-specific behavior behind a common LLM abstraction.
Expose safe traces for debugging and assessment observability.
Package the application with Docker for reproducible deployment.
7. Known Limitations
Provider-level malformed structured output may fail before semantic plan
repair can execute.
YouTube transcript availability depends on captions provided by YouTube and
the transcript API.
Free-tier deployment can experience cold starts after inactivity.
Audio transcription can be computationally expensive on limited instances.
OCR accuracy depends on source image quality and document layout.
The system does not provide unrestricted browsing or arbitrary URL fetching.
8. Future Improvements
Add bounded retries for provider-level structured-output failures.
Add stronger planner tool/reference compatibility validation.
Add persistent workflow checkpoints where appropriate.
Add streaming progress updates.
Add background processing for large audio and document inputs.
Add richer deployment metrics and observability.
Add rate limiting and authentication for production use.
Add automated CI/CD deployment verification.