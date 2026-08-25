"""GAIA question #9: from a grocery list, output only the items that are
botanically vegetables, alphabetized. Pure reasoning, no tools.
"""
from __future__ import annotations

from llama_index.core.llms import LLM

from gaia.base import QuestionSolver

SYSTEM = (
    "You are a general AI assistant. Solve the task carefully, then finish with a "
    "line:\nFINAL ANSWER: <answer>"
)


class Q9(QuestionSolver):
    number = 9

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        resp = await llm.acomplete(SYSTEM + "\n\n" + question.question)
        return str(resp)


if __name__ == "__main__":
    print(Q9().resolve())
