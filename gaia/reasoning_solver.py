"""A generic solver for pure-reasoning questions (no tools).

Uses a lean prompt on purpose: the big tool-oriented SYSTEM_PROMPT confuses the
model on self-contained puzzles (e.g. the reversed-sentence question, where it
replied "please provide the question" instead of solving it).
"""
from __future__ import annotations

from llama_index.core.llms import LLM

from gaia.base import QuestionSolver

REASONING_PROMPT = (
    "You are a general AI assistant. Solve the task using reasoning alone. "
    "Finish with a line:\nFINAL ANSWER: <answer>\n"
    "The answer must be as few words as possible, no articles, no punctuation, "
    "and formatted exactly as the question asks."
)


class ReasoningSolver(QuestionSolver):
    """Answers a reasoning question. Pass the number.

    `hint` is an optional task-clarification note (e.g. a domain rule) appended to
    the question — never the answer itself.
    """

    def __init__(self, number: int, hint: str | None = None) -> None:
        self.number = number
        self.hint = hint

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        prompt = REASONING_PROMPT + "\n\n" + question.question
        if self.hint:
            prompt += f"\n\nNote: {self.hint}"
        resp = await llm.acomplete(prompt)
        return str(resp)
