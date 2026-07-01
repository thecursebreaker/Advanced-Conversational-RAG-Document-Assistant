import re
from difflib import get_close_matches

from config import (
    MAX_TYPO_CORRECTIONS_PER_QUERY,
    TYPO_MIN_WORD_LENGTH,
    TYPO_SIMILARITY_CUTOFF,
)


def help_message() -> str:
    return (
        "You can ask:\n"
        "- document questions, e.g. 'who wrote the report?'\n"
        "- file/meta questions, e.g. 'what files do you have?'\n"
        "- file position questions, e.g. 'what is the second document about?'\n"
        "- memory questions, e.g. 'what did I ask 3 questions ago?'\n"
        "- control commands: help, reset, exit"
    )


def list_files(metadata: dict, max_files: int = 20) -> str:
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

    for f in files[:max_files]:
        lines.append(
            f"- {f['filename']} ({f['suffix']}, {f['chunks']} chunks)"
        )

    if len(files) > max_files:
        lines.append("")
        lines.append(f"...and {len(files) - max_files} more file(s).")

    return "\n".join(lines)


def ordinal_to_number(text: str):
    lowered = text.lower()

    mapping = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }

    for word, number in mapping.items():
        if word in lowered:
            return number

    match = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", lowered)
    if match:
        return int(match.group(1))

    return None


def get_file_by_position(metadata: dict, index: int):
    files = metadata.get("files", [])
    if index < 1 or index > len(files):
        return None
    return files[index - 1]


def describe_file_by_position(metadata: dict, index: int) -> str:
    file_info = get_file_by_position(metadata, index)
    if not file_info:
        return f"I could not find file number {index}."

    return (
        f"File {index} is {file_info['filename']} "
        f"({file_info['suffix']}, {file_info['chunks']} chunks)."
    )


def extract_file_position(question: str):
    if "document" not in question.lower():
        return None
    return ordinal_to_number(question)


def answer_memory_question(question: str, memory) -> str:
    q = question.lower().strip()

    if (
        "how many questions" in q
        or "how many things have i asked" in q
    ):
        count = memory.count_user_questions()
        return f"You have asked {count} question(s) so far."

    if "first thing i asked" in q or "first question" in q:
        first = memory.get_first_user_question()
        if first is None:
            return "You have not asked any questions yet."
        return f'The first thing you asked was: "{first}"'

    match = re.search(r"(\d+)\s+questions?\s+ago", q)
    if match:
        n = int(match.group(1))
        if n <= 0:
            return "Please ask for 1 or more questions ago."

        prev = memory.get_n_questions_ago(n)
        if prev is None:
            return f"You have not asked {n} question(s) yet."
        return f'{n} question(s) ago, you asked: "{prev}"'

    if "recent questions" in q or "what did i ask recently" in q:
        recent = memory.get_recent_questions()
        if not recent:
            return "You have not asked any questions yet."

        lines = ["Your recent questions were:"]
        for i, item in enumerate(recent, start=1):
            lines.append(f"{i}. {item}")
        return "\n".join(lines)

    return "I could not interpret that memory question."


def build_known_terms(metadata: dict) -> set[str]:
    files = metadata.get("files", [])
    vocab = set()

    for file_info in files:
        filename = file_info.get("filename", "").lower()

        cleaned = filename.replace(".pdf", "").replace(".docx", "").replace(".txt", "")
        cleaned = cleaned.replace("_", " ").replace("-", " ")

        words = re.findall(r"[a-zA-Z][a-zA-Z0-9\.\+]+", cleaned)
        for word in words:
            if len(word) >= TYPO_MIN_WORD_LENGTH:
                vocab.add(word.lower())

    vocab.update(
        {
            "linux",
            "arch",
            "ubuntu",
            "fedora",
            "debian",
            "mint",
            "kernel",
            "firefox",
            "thunderbird",
            "libreoffice",
            "gnome",
            "kde",
        }
    )

    return vocab


def fuzzy_correct_query(question: str, metadata: dict) -> str:
    vocab = build_known_terms(metadata)
    words = question.split()

    corrected_words = []
    corrections_used = 0

    for word in words:
        stripped = re.sub(r"[^a-zA-Z0-9]", "", word)
        lower = stripped.lower()

        if (
            len(lower) < TYPO_MIN_WORD_LENGTH
            or corrections_used >= MAX_TYPO_CORRECTIONS_PER_QUERY
            or lower in vocab
        ):
            corrected_words.append(word)
            continue

        matches = get_close_matches(
            lower,
            list(vocab),
            n=1,
            cutoff=TYPO_SIMILARITY_CUTOFF,
        )

        if matches:
            best = matches[0]
            corrected_words.append(best)
            corrections_used += 1
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


def score_query_variant(query: str, metadata: dict) -> int:
    vocab = build_known_terms(metadata)
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())

    score = 0

    for word in words:
        if word in vocab:
            score += 2

    if "arch" in words and "linux" in words:
        score += 5

    if "linus" in words:
        score -= 4

    if "arx" in words:
        score -= 2

    return score


def filter_query_variants(queries: list[str], metadata: dict) -> list[str]:
    unique = []
    for q in queries:
        if q and q not in unique:
            unique.append(q)

    scored = [(q, score_query_variant(q, metadata)) for q in unique]
    scored.sort(key=lambda item: item[1], reverse=True)

    best_score = scored[0][1] if scored else 0

    filtered = []
    for q, score in scored:
        if score >= best_score - 3:
            filtered.append(q)

    return filtered[:3]