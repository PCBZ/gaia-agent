"""A generic solver for web-retrieval questions.
"""
from __future__ import annotations

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from gaia.base import SYSTEM_PROMPT, QuestionSolver
from gaia.web import web_tools


class WebSolver(QuestionSolver):
    """Answers a web-retrieval question via agent + web_tools. Pass the number."""

    def __init__(self, number: int) -> None:
        self.number = number

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        agent = FunctionAgent(tools=web_tools(), llm=llm, system_prompt=SYSTEM_PROMPT)
        resp = await agent.run(user_msg=question.question)  # pyright: ignore[reportDeprecated]
        return str(resp)
