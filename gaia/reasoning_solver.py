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
    """Answers a reasoning question. Pass the number."""

    def __init__(self, number: int) -> None:
        self.number = number

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        resp = await llm.acomplete(REASONING_PROMPT + "\n\n" + question.question)
        return str(resp)
