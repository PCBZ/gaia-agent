"""GAIA question #20
"""
from __future__ import annotations

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from gaia.base import SYSTEM_PROMPT, QuestionSolver, openrouter_llm
from gaia.web import web_tools

SYSTEM = SYSTEM_PROMPT

class Q20(QuestionSolver):
    number = 20

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        agent = FunctionAgent(tools=web_tools(), llm=llm, system_prompt=SYSTEM)
        resp = await agent.run(user_msg=question.question)  # pyright: ignore[reportDeprecated]
        return str(resp)

if __name__ == "__main__":
    print(Q20().resolve(llm=openrouter_llm()))


