"""GAIA question #3: the reversed-sentence question. Pure reasoning, no tools."""
from __future__ import annotations

from llama_index.core.llms import LLM

from gaia.base import QuestionSolver

SYSTEM = (
    "You are a general AI assistant. Solve the task, then finish with a line:\n"
    "FINAL ANSWER: <answer>\n"
    "The answer must be as few words as possible, no articles, no punctuation."
)


class Q3(QuestionSolver):
    number = 3

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        resp = await llm.acomplete(SYSTEM + "\n\n" + question.question)
        return str(resp)


if __name__ == "__main__":
    print(Q3().resolve())
