Fully local Retrieval-Augmented Generation (RAG) system powered by Qwen via Ollama, designed for document-grounded question answering and corpus-level analysis.

Implements semantic retrieval using ChromaDB and Sentence Transformers, with automatic ingestion and indexing of PDF, DOCX, TXT, and Markdown files.

Features intelligent retrieval through multi-query expansion, typo-tolerant normalization, and relevance gating to improve robustness and reduce noise.

Supports multiple query intents including document QA, corpus-level summarization, file-specific lookup, and conversational memory-based queries.

Includes session-based conversational memory for follow-up questions and context-aware interactions.

Implements strict context-grounded answering with controlled refusal behavior to prevent hallucinations.

Incorporates evaluation framework with route accuracy tracking, keyword-based scoring, LLM-based answer grading, and failure analysis.

Designed with a modular and extensible architecture, enabling easy integration of new retrieval strategies, reranking components, or language models.
