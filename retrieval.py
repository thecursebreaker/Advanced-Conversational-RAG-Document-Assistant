import time

import chromadb
from chromadb.utils import embedding_functions

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_HISTORY_TURNS,
    MULTI_QUERY_FINAL_K,
    MULTI_QUERY_MAX_QUERIES,
    MULTI_QUERY_PER_QUERY_K,
)
from llm_utils import (
    answer_from_context,
    generate_query_variants,
    judge_relevance,
    summarize_document_context,
)
from tools import filter_query_variants, fuzzy_correct_query


REFUSAL_TEXT = "I could not find that in the provided documents."


def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    return collection


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


def retrieve_single_query(collection, query: str, k: int):
    start = time.perf_counter()
    results = collection.query(
        query_texts=[query],
        n_results=k,
    )
    elapsed = time.perf_counter() - start
    return results, elapsed


def merge_multi_query_results(results_list, final_k: int):
    merged_docs = []
    merged_metas = []
    seen = set()

    for result in results_list:
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]

        for doc, meta in zip(docs, metas):
            key = (
                meta.get("source", ""),
                meta.get("chunk_index", ""),
            )
            if key in seen:
                continue

            seen.add(key)
            merged_docs.append(doc)
            merged_metas.append(meta)

            if len(merged_docs) >= final_k:
                return {
                    "documents": [merged_docs],
                    "metadatas": [merged_metas],
                }

    return {
        "documents": [merged_docs],
        "metadatas": [merged_metas],
    }


def retrieve_multi_query(collection, queries: list[str]):
    all_results = []
    total_time = 0.0

    trimmed_queries = queries[:MULTI_QUERY_MAX_QUERIES]

    for query in trimmed_queries:
        result, elapsed = retrieve_single_query(
            collection=collection,
            query=query,
            k=MULTI_QUERY_PER_QUERY_K,
        )
        total_time += elapsed
        all_results.append(result)

    merged = merge_multi_query_results(
        results_list=all_results,
        final_k=MULTI_QUERY_FINAL_K,
    )

    return merged, total_time


def retrieve_file_chunks(collection, filename: str, k: int = 4):
    start = time.perf_counter()
    results = collection.get(
        where={"filename": filename},
        include=["documents", "metadatas"],
    )
    elapsed = time.perf_counter() - start

    docs = results.get("documents", [])[:k]
    metas = results.get("metadatas", [])[:k]

    return {
        "documents": [docs],
        "metadatas": [metas],
    }, elapsed


def handle_document_question(
    llm,
    memory,
    collection,
    question: str,
    logger,
    metadata: dict,
):
    history_text = memory.format_recent_history(MAX_HISTORY_TURNS)

    fuzzy_corrected_question = fuzzy_correct_query(
        question=question,
        metadata=metadata,
    )

    raw_query_variants = generate_query_variants(
        llm=llm,
        history_text=history_text,
        question=question,
        fuzzy_corrected_question=fuzzy_corrected_question,
    )

    query_variants = filter_query_variants(
        queries=raw_query_variants,
        metadata=metadata,
    )

    logger.info("fuzzy_corrected_question=%s", fuzzy_corrected_question)
    logger.info("raw_query_variants=%s", raw_query_variants)
    logger.info("filtered_query_variants=%s", query_variants)

    results, retrieval_time = retrieve_multi_query(
        collection=collection,
        queries=query_variants,
    )

    logger.info("multi_query_retrieval_time_seconds=%.4f", retrieval_time)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        logger.info("no_documents_retrieved=true")
        return {
            "answer": REFUSAL_TEXT,
            "query_variants": query_variants,
            "results": results,
            "relevant": False,
        }

    context = build_context(results)

    normalized_question = query_variants[0] if query_variants else question
    logger.info("normalized_question=%s", normalized_question)

    relevant = judge_relevance(
        llm=llm,
        question=normalized_question,
        context=context,
    )
    logger.info("relevance=%s", relevant)

    if not relevant:
        return {
            "answer": REFUSAL_TEXT,
            "query_variants": query_variants,
            "results": results,
            "relevant": False,
        }

    answer = answer_from_context(
        llm=llm,
        history_text=history_text,
        question=normalized_question,
        context=context,
    )

    if REFUSAL_TEXT.lower() in answer.lower():
        answer = REFUSAL_TEXT

    logger.info("retrieved_sources=%s", metas)

    return {
        "answer": answer,
        "query_variants": query_variants,
        "results": results,
        "relevant": True,
    }


def handle_file_summary_question(llm, filename: str, collection, logger):
    results, retrieval_time = retrieve_file_chunks(
        collection=collection,
        filename=filename,
        k=4,
    )

    logger.info("file_summary_filename=%s", filename)
    logger.info("file_summary_retrieval_time_seconds=%.4f", retrieval_time)

    docs = results.get("documents", [[]])[0]
    if not docs:
        return {
            "answer": "I could not retrieve content for that file.",
            "results": results,
        }

    context = build_context(results)
    answer = summarize_document_context(llm, context)

    return {
        "answer": answer,
        "results": results,
    }