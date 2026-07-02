import json
from pathlib import Path

from config import META_FILE, STRICT_DOC_MODE
from llm_utils import get_llm, small_chat_reply
from logger import get_logger
from memory import SessionMemory
from retrieval import get_collection, handle_document_question
from router import classify_route
from tools import answer_memory_question, list_files
from collections import Counter
from retrieval import handle_corpus_summary_question

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

def keyword_score(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0

    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)

def judge_answer(llm, question: str, answer: str) -> str:
    prompt = f"""
You are evaluating an AI assistant.

Question:
{question}

Answer:
{answer}

Is this answer:
- GOOD (correct and relevant)
- PARTIAL (somewhat correct)
- BAD (incorrect or irrelevant)

Respond with only one word: GOOD, PARTIAL, or BAD.
"""
    result = llm.invoke(prompt) or ""
    result = (llm.invoke(prompt) or "").strip().upper()

    if "GOOD" in result:
        return "GOOD"
    elif "PARTIAL" in result:
        return "PARTIAL"
    else:
        return "BAD

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
    
    total_keyword_score = 0
    good = 0
    partial = 0
    bad = 0
    refusals = 0
    route_counter = Counter()
    failures = []
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
        elif actual_route == "corpus_summary":        
            result = handle_corpus_summary_question(
                llm=llm,
                memory=memory,
                collection=collection,
                question=question,
                logger=logger,
                metadata=metadata,
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
        expected_keywords = test.get("expected_keywords", [])
        score = keyword_score(answer, expected_keywords)
        total_keyword_score += score
        print(f"Keyword score: {score:.2f}")
        judgment = judge_answer(llm, question, answer)
        print(f"LLM judgment: {judgment}")
        if judgment == "GOOD": good += 1
        elif judgment == "PARTIAL": partial += 1
        else: bad += 1
        if "could not find" in answer.lower():
            refusals += 1
        print(f"Running refusals: {refusals}")
        print("-" * 50)
        is_refusal = "i could not find" in answer.lower()
        answer_length = len(answer.split())
        print(f"Refusal: {is_refusal}")
        print(f"Answer length: {answer_length}")
        route_counter[actual_route] += 1
        if not route_ok:
            failures.append((question, expected_route, actual_route))

    route_accuracy = route_correct / total if total else 0
    print(f"Total tests: {total}")
    print(f"Route accuracy: {route_accuracy:.2%}")
    avg_score = total_keyword_score / total if total else 0
    print(f"Average keyword score: {avg_score:.2f}")
    print(f"GOOD: {good}")
    print(f"PARTIAL: {partial}")
    print(f"BAD: {bad}")
    print(f"Refusals: {refusals}/{total}")
    print("\nRoute distribution:")
    for route, count in route_counter.items():
        print(f"{route}: {count}")
    print("\nFailures:")
    for q, exp, act in failures:
        print(f"Q: {q}")
        print(f"Expected: {exp}, Got: {act}")
        print("-" * 30)    

if __name__ == "__main__":
    main()
