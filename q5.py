"""GAIA question #5: who nominated the only English Wikipedia Featured Article
about a dinosaur that was promoted in November 2016? Web retrieval (Wikipedia
FA logs + the FAC nomination page).
"""
from __future__ import annotations

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from gaia.base import SYSTEM_PROMPT, QuestionSolver, huggingface_llm
from gaia.web import web_tools

SYSTEM = SYSTEM_PROMPT


class Q5(QuestionSolver):
    number = 5

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        agent = FunctionAgent(tools=web_tools(), llm=llm, system_prompt=SYSTEM)
        resp = await agent.run(user_msg=question.question)  # pyright: ignore[reportDeprecated]
        return str(resp)


if __name__ == "__main__":
    print(Q5().resolve(llm=huggingface_llm()))
