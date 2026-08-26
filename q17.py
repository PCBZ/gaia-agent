"""GAIA question #17: which country had the fewest athletes at the 1928 Summer
Olympics? Tie -> first alphabetically. Answer = IOC country code. Web retrieval.

Several nations sent a single athlete, so the alphabetical tie-break matters.
"""
from __future__ import annotations

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM

from gaia.base import QuestionSolver, huggingface_llm
from gaia.web import web_tools

SYSTEM = (
    "Use the web_search and read_url tools to look up how many athletes each "
    "country sent. Do not assume a country is small — verify its athlete count "
    "before concluding. Find the smallest count, list every country tied at that "
    "count, then choose the one that comes first alphabetically. Answer with that "
    "country's IOC code only. Finish with a line:\nFINAL ANSWER: <code>"
)


class Q17(QuestionSolver):
    number = 17

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        agent = FunctionAgent(tools=web_tools(), llm=llm, system_prompt=SYSTEM)
        resp = await agent.run(user_msg=question.question)  # pyright: ignore[reportDeprecated]
        return str(resp)


if __name__ == "__main__":
    print(Q17().resolve(llm=huggingface_llm()))
