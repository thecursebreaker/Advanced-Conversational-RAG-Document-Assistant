import json
from pathlib import Path

from config import META_FILE, STRICT_DOC_MODE
from llm_utils import get_llm, small_chat_reply
from logger import get_logger
from memory import SessionMemory
from retrieval import get_collection, handle_document_question
from router import classify_route
from tools import answer_memory_question, list_files


EVAL_FILE = Path("tests/eval_questions.json")


def load_metadata():
    if not META_FILE.exists():
        return {"files": [], "file_count": 0}

    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"files": [], "file_count": 0}


def handle_general():
    if STRICT_DOC_MODE:
        return (
            "That question is outside the scope of the indexed documents. "
            "Ask me about your files instead."
        )
    return "General mode is disabled in this project version."


def main():
    if not EVAL_FILE.exists():
        print("Missing tests/eval_questions.json")
        return

    logger = get_logger("eval")
    llm = get_llm()
    collection = get_collection()
    metadata = load_metadata()
    memory = SessionMemory()

    tests = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

    total = len(tests)
    route_correct = 0

    for i, test in enumerate(tests, start=1):
        question = test["question"]
        expected_route = test["expected_route"]

        actual_route = classify_route(question)
        route_ok = actual_route == expected_route
        if route_ok:
            route_correct += 1

        if actual_route == "chat":
            answer = small_chat_reply(question)
        elif actual_route == "meta":
            answer = list_files(metadata)
        elif actual_route == "memory":
            answer = answer_memory_question(question, memory)
        elif actual_route == "general":
            answer = handle_general()
        elif actual_route == "document":
            result = handle_document_question(
                llm=llm,
                memory=memory,
                collection=collection,
                question=question,
                logger=logger,
            )
            answer = result["answer"]
        else:
            answer = "Unhandled route."

        memory.add_turn(question, answer)

        print(f"Test {i}")
        print(f"Q: {question}")
        print(f"Expected route: {expected_route}")
        print(f"Actual route:   {actual_route}")
        print(f"Route correct:  {route_ok}")
        print(f"A: {answer}")
        print("-" * 50)

    route_accuracy = route_correct / total if total else 0
    print(f"Total tests: {total}")
    print(f"Route accuracy: {route_accuracy:.2%}")


if __name__ == "__main__":
    main()
