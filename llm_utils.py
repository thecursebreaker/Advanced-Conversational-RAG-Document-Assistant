from langchain_ollama import OllamaLLM

from config import OLLAMA_MODEL


def get_llm():
    return OllamaLLM(model=OLLAMA_MODEL)


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

Examples:
User question: What year?
Chat history implies Linux Mint release.
Good output:
What year was the Linux Mint release?

User question: When is Arx Linux available for download?
Good output:
When is Arx Linux available for download?
When is Arch Linux available for download?

User question: summarize it
Chat history implies the second document.
Good output:
Summarize the second document.

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
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)

    if fuzzy_corrected_question and fuzzy_corrected_question not in cleaned:
        cleaned.append(fuzzy_corrected_question)

    if question not in cleaned:
        cleaned.insert(0, question)

    unique = []
    for q in cleaned:
        if q and q not in unique:
            unique.append(q)

    return unique[:3]


def judge_relevance(llm, question: str, context: str) -> bool:
    prompt = f"""
You are a retrieval relevance judge for a document-grounded assistant.

Task:
Decide whether the retrieved context contains enough evidence to answer the
user's intended question.

Important guidance:
- The context may contain both relevant and irrelevant retrieved chunks.
- Answer YES if at least one part of the context provides enough evidence to
  answer the intended question.
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
- If the answer is not supported by the context, output exactly:
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