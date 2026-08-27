"""GAIA question #7: in the linked YouTube video, what does Teal'c say in response
to "Isn't that hot?"

Expected answer: Extremely.
"""
from __future__ import annotations

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from gaia.base import SYSTEM_PROMPT, QuestionSolver, openrouter_llm
from gaia.media import media_tools


class Q7(QuestionSolver):
    number = 7

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        agent = FunctionAgent(tools=media_tools(), llm=llm, system_prompt=SYSTEM_PROMPT)
        resp = await agent.run(user_msg=question.question)  # pyright: ignore[reportDeprecated]
        return str(resp)


if __name__ == "__main__":
    print(Q7().resolve(llm=openrouter_llm()))
