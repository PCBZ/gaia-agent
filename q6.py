"""GAIA question #6: find the subset of S in any counter-example that proves the
operation * is not commutative. Pure reasoning, no tools.

Expected answer: b, e  (only b*e != e*b in the given table).
"""
from __future__ import annotations

from llama_index.core.llms import LLM

from gaia.base import QuestionSolver
from gaia.questions import load_questions

SYSTEM = (
    "You are a general AI assistant. Solve the task carefully, then finish with a "
    "line:\nFINAL ANSWER: <answer>"
)


class Q6(QuestionSolver):
    number = 6
    index = 5

    async def solve(self, llm: LLM) -> str:
        question = load_questions()[self.index]
        print(f"Question {self.number}: {question.question}")
        resp = await llm.acomplete(SYSTEM + "\n\n" + question.question)
        return str(resp)


if __name__ == "__main__":
    print(Q6().resolve())
