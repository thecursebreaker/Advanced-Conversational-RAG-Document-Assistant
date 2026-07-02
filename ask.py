import json

from config import (
    META_FILE,
    SHOW_QUERY_VARIANTS,
    SHOW_RETRIEVED_SOURCES,
    STRICT_DOC_MODE,
)
from llm_utils import get_llm, small_chat_reply
from logger import get_logger
from memory import SessionMemory
from retrieval import (
    get_collection,
    handle_corpus_summary_question,
    handle_document_question,
    handle_file_summary_question,
)
from router import classify_route
from tools import (
    answer_memory_question,
    extract_file_position,
    get_file_by_position,
    help_message,
    list_files,
)


def load_metadata():
    if not META_FILE.exists():
        return {"files": [], "file_count": 0}

    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"files": [], "file_count": 0}


def handle_control(question: str, memory: SessionMemory):
    q = question.lower().strip()

    if q in {"exit", "quit"}:
        return {"action": "exit", "answer": "Goodbye."}

    if q == "reset":
        memory.reset()
        return {"action": "continue", "answer": "Chat memory cleared."}

    if q == "help":
        return {"action": "continue", "answer": help_message()}

    return {"action": "continue", "answer": "Unknown control command."}


def handle_general(question: str):
    if STRICT_DOC_MODE:
        return (
            "That question is outside the scope of the indexed documents. "
            "Ask me about your files instead."
        )
    return "General mode is disabled in this project version."


def main():
    logger = get_logger()
    logger.info("application_start=true")

    metadata = load_metadata()
    llm = get_llm()
    collection = get_collection()
    memory = SessionMemory()

    print("LLM document system is ready.")
    print("Type 'help' for examples.")
    print("Type 'reset' to clear chat memory.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue

        logger.info("USER: %s", question)

        route = classify_route(question)
        logger.info("route=%s", route)

        if route == "control":
            result = handle_control(question, memory)
            answer = result["answer"]

            print("\nAssistant:")
            print(answer)

            logger.info("ASSISTANT: %s", answer)

            if question.lower().strip() not in {"reset"}:
                memory.add_turn(question, answer)

            if result["action"] == "exit":
                logger.info("application_exit=true")
                break

            continue

        if route == "chat":
            answer = small_chat_reply(question)

            print("\nAssistant:")
            print(answer)

            logger.info("ASSISTANT: %s", answer)

            memory.add_turn(question, answer)
            continue

        if route == "meta":
            answer = list_files(metadata)

            print("\nAssistant:")
            print(answer)

            logger.info("ASSISTANT: %s", answer)

            memory.add_turn(question, answer)
            continue

        if route == "memory":
            answer = answer_memory_question(question, memory)

            print("\nAssistant:")
            print(answer)

            logger.info("ASSISTANT: %s", answer)

            memory.add_turn(question, answer)
            continue

        if route == "general":
            answer = handle_general(question)

            print("\nAssistant:")
            print(answer)

            logger.info("ASSISTANT: %s", answer)

            memory.add_turn(question, answer)
            continue

        if route == "corpus_summary":
            result = handle_corpus_summary_question(
                llm=llm,
                memory=memory,
                collection=collection,
                question=question,
                logger=logger,
                metadata=metadata,
            )

            answer = result["answer"]

            if SHOW_QUERY_VARIANTS:
                print("\nQuery variants used:")
                for i, query in enumerate(result["query_variants"], start=1):
                    print(f"{i}. {query}")

            print("\nAssistant:")
            print(answer)

            logger.info("ASSISTANT: %s", answer)

            if SHOW_RETRIEVED_SOURCES:
                print("\nRetrieved sources:")
                for i, meta in enumerate(result["results"]["metadatas"][0], start=1):
                    print(f"{i}. {meta}")

            memory.add_turn(question, answer)
            continue
        
        if route == "file_lookup":
            position = extract_file_position(question)

            if position is None:
                answer = "I could not determine which document number you meant."

                print("\nAssistant:")
                print(answer)

                logger.info("ASSISTANT: %s", answer)

                memory.add_turn(question, answer)
                continue

            file_info = get_file_by_position(metadata, position)

            if file_info is None:
                answer = f"I could not find file number {position}."

                print("\nAssistant:")
                print(answer)

                logger.info("ASSISTANT: %s", answer)

                memory.add_turn(question, answer)
                continue

            result = handle_file_summary_question(
                llm=llm,
                filename=file_info["filename"],
                collection=collection,
                logger=logger,
            )

            answer = (
                f"Document {position} is {file_info['filename']}.\n\n"
                f"{result['answer']}"
            )

            print("\nAssistant:")
            print(answer)

            logger.info("ASSISTANT: %s", answer)

            if SHOW_RETRIEVED_SOURCES:
                print("\nRetrieved sources:")
                for i, meta in enumerate(result["results"]["metadatas"][0], start=1):
                    print(f"{i}. {meta}")

            memory.add_turn(question, answer)
            continue

        result = handle_document_question(
            llm=llm,
            memory=memory,
            collection=collection,
            question=question,
            logger=logger,
            metadata=metadata,
        )

        answer = result["answer"]

        if SHOW_QUERY_VARIANTS:
            print("\nQuery variants used:")
            for i, query in enumerate(result["query_variants"], start=1):
                print(f"{i}. {query}")

        print("\nAssistant:")
        print(answer)

        logger.info("ASSISTANT: %s", answer)

        if SHOW_RETRIEVED_SOURCES:
            print("\nRetrieved sources:")
            for i, meta in enumerate(result["results"]["metadatas"][0], start=1):
                print(f"{i}. {meta}")

        memory.add_turn(question, answer)


if __name__ == "__main__":
    main()