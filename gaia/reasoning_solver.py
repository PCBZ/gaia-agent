"""A generic solver for reasoning questions.
"""
from __future__ import annotations

from llama_index.core.llms import LLM
from gaia.base import SYSTEM_PROMPT, QuestionSolver



class ReasoningSolver(QuestionSolver):
    """Answers a reasoning question. Pass the number."""

    def __init__(self, number: int) -> None:
        self.number = number

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        resp = await llm.acomplete(SYSTEM_PROMPT + "\n\n" + question.question)
        return str(resp)
