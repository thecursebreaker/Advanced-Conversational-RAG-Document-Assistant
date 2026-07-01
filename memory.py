class SessionMemory:
    def __init__(self):
        self.turns = []

    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append({"user": user, "assistant": assistant})

    def reset(self) -> None:
        self.turns = []

    def count_user_questions(self) -> int:
        return len(self.turns)

    def get_first_user_question(self):
        if not self.turns:
            return None
        return self.turns[0]["user"]

    def get_n_questions_ago(self, n: int):
        if n <= 0:
            return None

        questions = [turn["user"] for turn in self.turns]
        if len(questions) < n:
            return None

        return questions[-n]

    def get_recent_questions(self, limit: int = 5):
        questions = [turn["user"] for turn in self.turns]
        return questions[-limit:]

    def format_recent_history(self, max_turns: int = 6) -> str:
        recent = self.turns[-max_turns:]
        lines = []

        for turn in recent:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")

        return "\n".join(lines)
