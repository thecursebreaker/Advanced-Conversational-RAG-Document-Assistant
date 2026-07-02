from langchain_ollama import OllamaLLM

from config import OLLAMA_MODEL

import re
from sentence_transformers import SentenceTransformer, util
import torch


semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

KNOWN_TERMS = [
    "arch linux",
    "linux",
    "ubuntu",
    "kde",
    "kali linux",
    "kernel",
    "release",
    "download",
]

KNOWN_EMBEDDINGS = semantic_model.encode(KNOWN_TERMS, convert_to_tensor=True)


def semantic_correct_query(query: str, threshold: float = 0.75) -> str:
    query_emb = semantic_model.encode(query, convert_to_tensor=True)

    similarities = util.cos_sim(query_emb, KNOWN_EMBEDDINGS)[0]
    best_idx = torch.argmax(similarities).item()
    best_score = similarities[best_idx].item()

    if best_score > threshold:
        return query + " " + KNOWN_TERMS[best_idx]

    return query


def get_llm():
    return OllamaLLM(model=OLLAMA_MODEL)

def is_valid_query(q: str) -> bool:

    if re.search(r"[a-zA-Z]+\d{3,}", q):
        return False

    if re.search(r"\b[a-zA-Z0-9]{8,}\b", q):
        if not re.search(r"[a-zA-Z]{4,}", q):
            return False

    return True

def generate_query_variants(
    llm,
    history_text: str,
    question: str,
    fuzzy_corrected_question: str | None = None,
) -> list[str]:
    typo_hint = ""
    if fuzzy_corrected_question and fuzzy_corrected_question.strip() != question.strip():
        typo_hint = f"""
Possible typo-normalized version:
{fuzzy_corrected_question}
""".strip()

    prompt = f"""
You are a retrieval query optimizer for a document-grounded assistant.

Your task is to generate up to 3 high-quality semantic retrieval queries.

Objectives:
1. Preserve the user's intended meaning exactly.
2. Resolve conversational references using chat history when needed.
3. Correct obvious spelling mistakes or typos when strongly supported.
4. Improve retrieval robustness without changing the scope of the question.

Rules:
- Query 1 must be a standalone version of the user's question.
- Additional queries may be corrected or normalized variants if they preserve
  the same meaning and improve retrieval.
- Prefer obvious typo corrections only when strongly justified.
- Do not broaden, narrow, or reinterpret the question.
- Do not answer the question.
- Do not explain anything.
- Output only the final queries, one per line.
- If only 1 or 2 queries are useful, output only those.
- NEVER modify numbers (e.g. years must stay exactly the same)

Chat history:
{history_text}

Latest user question:
{question}

{typo_hint}

Search queries:
""".strip()

    result = llm.invoke(prompt) or ""
    raw_lines = [line.strip() for line in result.splitlines() if line.strip()]

    cleaned = []
    for line in raw_lines:
        normalized = line.lstrip("-•*0123456789. ").strip()

        if not normalized:
            continue

        if not is_valid_query(normalized):
            continue

        if normalized not in cleaned:
            cleaned.append(normalized)

    # Always include corrected version FIRST
    if fuzzy_corrected_question:
        if fuzzy_corrected_question in cleaned:
            cleaned.remove(fuzzy_corrected_question)
        cleaned.insert(0, fuzzy_corrected_question)

    # Always include original LAST
    if question in cleaned:
        cleaned.remove(question)
    cleaned.append(question)

    # Deduplicate while preserving order
    unique = []
    for q in cleaned:
        if q and q not in unique:
            unique.append(q)

    # Final safety fallback
    if not unique:
        if fuzzy_corrected_question: return [fuzzy_corrected_question]
        return [question]

    return unique[:3]


def judge_relevance(llm, question: str, context: str) -> bool:
    prompt = f"""
You are a retrieval relevance judge for a document-grounded assistant.

Task:
Decide whether the retrieved context contains enough evidence to answer the
user's intended question.

Important guidance:
- The context may contain both relevant and irrelevant retrieved chunks.
Answer YES if the context contains ANY information that could help answer
the question, even partially.

Be permissive:
- If the context includes related events, releases, summaries, or relevant topics,
  answer YES.
- Do NOT require a complete or perfect answer.
- Only answer NO if the context is completely unrelated.
- Treat obvious minor typos or spelling mistakes in the question as acceptable
  if the intended meaning is clear.
- Answer NO only if the context does not provide enough evidence at all.
- Be reasonably cautious, but do not reject valid evidence because of some noise.
- Output only YES or NO.

Question:
{question}

Retrieved context:
{context}

Relevant?
""".strip()

    result = (llm.invoke(prompt) or "").strip().upper()
    return result.startswith("YES")


def answer_from_context(llm, history_text: str, question: str, context: str) -> str:
    prompt = f"""
You are a document-grounded question-answering assistant.

Your job is to answer the user's question using only the provided document
context.

Answering policy:
- Use the document context as the factual source of truth.
- Use chat history only to resolve references in the user's wording.
- Ignore irrelevant chunks if some chunks are useful.
- If at least one chunk contains enough evidence, answer from that evidence.
- Treat obvious minor typos in the user's wording as acceptable if the intended
  meaning is clear from the context.
- Do not use outside knowledge.
- If the answer is explicitly stated OR can be reasonably inferred from the context, answer it.
- If a date, release, or event is mentioned, use it to answer "when" questions.
- Only refuse if the answer is completely missing from the context.
- If refusing, output exactly:
I could not find that in the provided documents.
- Do not add extra explanation after that refusal.
- If supported, answer concisely and directly.
- Cite supporting evidence using references like [Source 1].
- Prefer the strongest evidence.

Recent chat history:
{history_text}

Document context:
{context}

Current user question:
{question}

Answer:
""".strip()

    answer = llm.invoke(prompt)
    return (answer or "").strip()


def summarize_document_context(llm, context: str) -> str:
    prompt = f"""
You are summarizing one indexed document.

Rules:
- Use only the provided document content.
- Summarize what the file is mainly about in 2 to 4 sentences.
- Mention key topics, events, or releases if clearly present.
- Do not use outside knowledge.

Document content:
{context}

Summary:
""".strip()

    answer = llm.invoke(prompt)
    return (answer or "").strip()


def summarize_corpus_context(llm, question: str, context: str) -> str:
    prompt = f"""
You are extracting events from document snippets.

Task:
Extract ALL meaningful events mentioned in the context.

Definition of "event":
- software release
- major update
- announcement
- notable change

STRICT RULES:
- Extract AT LEAST 5 events if available
- Each bullet point MUST represent a DIFFERENT event
- Do NOT merge multiple events into one
- Do NOT summarize broadly
- Do NOT write paragraphs
- ONLY output bullet points

Coverage requirement:
- The context contains multiple sources labeled [Source X]
- You MUST extract events from MULTIPLE different sources
- Do NOT focus on a single source

Formatting:
- Bullet list ONLY
- One event per bullet
- Cite sources like (Source 1)

If nothing meaningful is found, output exactly:
I could not find that in the provided documents.

Context:
{context}

Events:
""".strip()

    answer = llm.invoke(prompt)
    return (answer or "").strip()


def small_chat_reply(question: str) -> str:
    q = question.lower().strip()

    if q in {"hello", "hi", "hey"}:
        return (
            "Hello! You can ask me about your indexed documents, the files I "
            "have indexed, or your current chat session."
        )

    if q in {"thanks", "thank you"}:
        return "You're welcome."

    if q in {"good morning", "good evening"}:
        return "Hello! Ask me about your indexed documents whenever you're ready."

    return "Hello."