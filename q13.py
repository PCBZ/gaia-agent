"""GAIA question #13: how many at bats did the Yankee with the most walks in the
1977 regular season have that same season? Web retrieval.
"""
from __future__ import annotations

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from gaia.base import SYSTEM_PROMPT, QuestionSolver, huggingface_llm
from gaia.web import web_tools

SYSTEM = SYSTEM_PROMPT

class Q13(QuestionSolver):
    number = 13

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        agent = FunctionAgent(tools=web_tools(), llm=llm, system_prompt=SYSTEM)
        resp = await agent.run(user_msg=question.question)  # pyright: ignore[reportDeprecated]
        return str(resp)

if __name__ == "__main__":
    print(Q13().resolve(llm=huggingface_llm()))


