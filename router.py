import re
from tools import fuzzy_correct_query

def classify_route(question: str) -> str:
    q = question.lower().strip()
    
    normalized_q = fuzzy_correct_query(question, {"files": []}).lower().strip()
    
    if q in {"exit", "quit", "reset", "help"}:
        return "control"

    chat_patterns = {
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good evening",
    }
    if q in chat_patterns:
        return "chat"
    
    if re.search(r"\bhow m\w+ (documents|files)\b", normalized_q):
        return "meta"
    
    if re.search(r"\b(what|which)\b.*\b(documents|files)\b", normalized_q):
        return "meta"
    
    if re.search(r"\b\d+(?:st|nd|rd|th)\b", normalized_q):
        return "file_lookup"
    
    if re.search(r"\b(first|second|third|\d+(?:st|nd|rd|th)?)\s+document\b", normalized_q):
        return "file_lookup"

    if re.search(r"\b(what happened|events)\b", normalized_q):
        return "corpus_summary"

    if re.search(r"\b(summarize|summarise).*(documents|files|corpus)\b", normalized_q):
        return "corpus_summary"

    if re.search(r"\bwhat (topics|events).*(documents|files)\b", normalized_q):
        return "corpus_summary"
    

    if "what files" in q or "which files" in q or "what documents" in q:
        return "meta"

    meta_keywords = [
        "list files",
        "list the files",
        "show files",
        "what do you have",
        "what is indexed",
        "what do you know",
        "database",
    ]
    if any(keyword in normalized_q for keyword in meta_keywords):
        return "meta"

    memory_patterns = [
        r"what did i ask",
        r"how many questions have i asked",
        r"how many questions did i ask",
        r"how many questions.*so far",
        r"first thing i asked",
        r"first question",
        r"questions ago",
        r"recent questions",
    ]
    if any(re.search(pattern, normalized_q) for pattern in memory_patterns):
        return "memory"

    math_like = [
        r"^\d+\s*[\+\-\*/]\s*\d+\s*=?$",
    ]
    if any(re.match(pattern, normalized_q) for pattern in math_like):
        return "general"

    general_keywords = [
        "what is morality",
        "what is the universe made of",
        "what does being gay mean",
        "define morality",
    ]
    if any(keyword in normalized_q for keyword in general_keywords):
        return "general"

    return "document"