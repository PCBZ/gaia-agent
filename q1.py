"""GAIA question #1: how many studio albums did Mercedes Sosa release between
2000 and 2009 (inclusive)? Needs web retrieval.

Expected answer: 3.
"""
from __future__ import annotations

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from gaia.base import QuestionSolver
from gaia.web import web_tools

SYSTEM = (
    "Use the web_search and read_url tools to find the answer from the web. "
    "Finish with a line:\nFINAL ANSWER: <number>"
)


class Q1(QuestionSolver):
    number = 1

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        agent = FunctionAgent(tools=web_tools(), llm=llm, system_prompt=SYSTEM)
        resp = await agent.run(user_msg=question.question)  # pyright: ignore[reportDeprecated]
        return str(resp)


if __name__ == "__main__":
    print(Q1().resolve())
