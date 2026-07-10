# Multimodal Agentic AI Application

A deployed multimodal agentic AI application capable of processing text, PDFs, images, and audio files, understanding user intent, asking clarification questions when required, autonomously executing multi-step tool plans, and returning text-only results with extracted content and safe execution traces.

## Status

Under active development.

## Core Architecture

User Request
→ FastAPI
→ Secure Upload Gateway
→ Multimodal Extraction
→ Content Normalization
→ Deterministic URL Detection
→ LangGraph Agent Workflow
→ Validated Tool Execution
→ Response Composition

## Planned Capabilities

- Text input
- PDF text extraction
- Scanned-PDF OCR fallback
- Image OCR with confidence
- Audio transcription
- Multi-file processing
- Intent understanding
- Mandatory clarification handling
- Autonomous multi-tool chaining
- YouTube transcript retrieval
- Summarization
- Sentiment analysis
- Code explanation
- Cross-input comparison
- Conversational answering
- Safe execution traces
- Provider fallback
- Docker deployment
- Public deployment

## Tech Stack

- Python 3.11
- FastAPI
- Pydantic
- LangGraph
- LangChain
- Groq
- Gemini
- PyMuPDF
- Tesseract OCR
- faster-whisper
- Vanilla JavaScript
- Docker
- Render

## Development

Implementation follows phased milestones with tests and stable Git commits after each checkpoint.