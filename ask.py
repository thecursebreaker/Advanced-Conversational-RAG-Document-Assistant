import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaLLM


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "my_documents"
META_FILE = Path("collection_meta.json")
OLLAMA_MODEL = "qwen2.5:7b"


def load_metadata():
    if not META_FILE.exists():
        return {"files": [], "file_count": 0}

    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"files": [], "file_count": 0}


def is_meta_question(question: str) -> bool:
    q = question.lower().strip()

    patterns = [
        "what do you know",
        "what can you do",
        "what files",
        "which files",
        "what documents",
        "which documents",
        "what is in the database",
        "what do you have",
        "summarize the files",
        "list the files",
        "list files",
        "show files",
    ]

    return any(p in q for p in patterns)


def answer_meta_question(metadata: dict) -> str:
    files = metadata.get("files", [])
    if not files:
        return (
            "I do not have any indexed files yet. Put documents in the data "
            "folder and run ingest.py first."
        )

    lines = [
        f"I currently have access to {len(files)} indexed file(s).",
        "",
        "Files:",
    ]

    for f in files[:20]:
        lines.append(
            f"- {f['filename']} ({f['suffix']}, {f['chunks']} chunks)"
        )

    lines.append("")
    lines.append(
        "You can ask me questions about facts, dates, people, summaries, "
        "or topics found in those files."
    )

    return "\n".join(lines)


def format_chat_history(chat_history, max_turns=6) -> str:
    recent = chat_history[-max_turns:]
    lines = []

    for turn in recent:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")

    return "\n".join(lines)


def rewrite_question(llm, chat_history, question: str) -> str:
    history_text = format_chat_history(chat_history)

    prompt = f"""
You rewrite follow-up questions into clear standalone questions for
document retrieval.

Rules:
- Keep the meaning exactly the same.
- Use the chat history only to resolve references like "he", "it",
  "that year", "what about the second one", etc.
- If the user's question is already standalone, return it unchanged.
- Return only the rewritten question.
- Do not answer the question.

Chat history:
{history_text}

Latest user question:
{question}

Standalone question:
""".strip()

    rewritten = llm.invoke(prompt)
    return (rewritten or question).strip()


def build_context(results) -> str:
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        source = meta.get("source", "Unknown source")
        chunk_index = meta.get("chunk_index", "Unknown chunk")
        parts.append(
            f"[Source {i}: {source}, chunk {chunk_index}]\n{doc}"
        )

    return "\n\n".join(parts)


def answer_with_rag(llm, chat_history, question: str, context: str) -> str:
    history_text = format_chat_history(chat_history)

    prompt = f"""
You are a helpful assistant answering questions using only the provided
document context.

Rules:
- Use chat history only to understand references in the conversation.
- Use the document context as the factual source.
- If the answer is not in the context, say exactly:
  "I could not find that in the provided documents."
- Do not invent facts.
- Cite relevant sources like [Source 1].
- Be concise but complete.

Recent chat history:
{history_text}

Document context:
{context}

Current user question:
{question}

Answer:
""".strip()

    answer = llm.invoke(prompt)
    return answer.strip()


def main():
    metadata = load_metadata()

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    llm = OllamaLLM(model=OLLAMA_MODEL)

    chat_history = []

    print("Conversational local RAG is ready.")
    print("Type 'exit' to quit.")
    print("Type 'reset' to clear chat memory.")

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        if question.lower() == "reset":
            chat_history = []
            print("Assistant: Chat memory cleared.")
            continue

        if is_meta_question(question):
            answer = answer_meta_question(metadata)
            print("\nAssistant:")
            print(answer)

            chat_history.append(
                {
                    "user": question,
                    "assistant": answer,
                }
            )
            continue

        standalone_question = rewrite_question(
            llm=llm,
            chat_history=chat_history,
            question=question,
        )

        results = collection.query(
            query_texts=[standalone_question],
            n_results=4,
        )

        docs = results.get("documents", [[]])[0]
        if not docs:
            answer = "I could not find that in the provided documents."
            print("\nAssistant:")
            print(answer)

            chat_history.append(
                {
                    "user": question,
                    "assistant": answer,
                }
            )
            continue

        context = build_context(results)

        answer = answer_with_rag(
            llm=llm,
            chat_history=chat_history,
            question=question,
            context=context,
        )

        print("\nStandalone retrieval question:")
        print(standalone_question)

        print("\nAssistant:")
        print(answer)

        print("\nRetrieved sources:")
        for i, meta in enumerate(results["metadatas"][0], start=1):
            print(f"{i}. {meta}")

        chat_history.append(
            {
                "user": question,
                "assistant": answer,
            }
        )


if __name__ == "__main__":
    main()