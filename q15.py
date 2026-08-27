"""GAIA question #15
"""
from __future__ import annotations

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from gaia.base import SYSTEM_PROMPT, QuestionSolver, groq_llm, huggingface_llm, openrouter_llm
from gaia.web import web_tools

SYSTEM = SYSTEM_PROMPT

class Q15(QuestionSolver):
    number = 15

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        agent = FunctionAgent(tools=web_tools(), llm=llm, system_prompt=SYSTEM)
        resp = await agent.run(user_msg=question.question)  # pyright: ignore[reportDeprecated]
        return str(resp)

if __name__ == "__main__":
    print(Q15().resolve(llm=openrouter_llm()))


