"""GAIA question #17: which country had the fewest athletes at the 1928 Summer
Olympics? Tie -> first alphabetically. Answer = IOC country code. Web retrieval.

Several nations sent a single athlete, so the alphabetical tie-break matters.
"""
from __future__ import annotations

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from gaia.base import QuestionSolver
from gaia.web import web_tools

SYSTEM = (
    "Use the web_search and read_url tools. Several countries may be tied for the "
    "fewest athletes; enumerate all of them, then pick the first alphabetically. "
    "Answer with the IOC country code only. Finish with a line:\n"
    "FINAL ANSWER: <code>"
)


class Q17(QuestionSolver):
    number = 17

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        agent = FunctionAgent(tools=web_tools(), llm=llm, system_prompt=SYSTEM)
        resp = await agent.run(user_msg=question.question)  # pyright: ignore[reportDeprecated]
        return str(resp)


if __name__ == "__main__":
    print(Q17().resolve())
