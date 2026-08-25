"""GAIA question #19: from the attached Excel of fast-food sales, total the sales
from food (excluding drinks), in USD with two decimals.
"""
from __future__ import annotations

import pandas as pd
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec

from gaia.base import QuestionSolver
from gaia.questions import download_attachment

SYSTEM = (
    "Use the code_interpreter tool to compute the answer. "
    "Finish with a line:\nFINAL ANSWER: <number>"
)


class Q19(QuestionSolver):
    number = 19

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        path = download_attachment(question)
        table = pd.read_excel(path).to_string(index=False)

        agent = FunctionAgent(
            tools=CodeInterpreterToolSpec().to_tool_list(),
            llm=llm,
            system_prompt=SYSTEM,
        )
        prompt = f"{question.question}\n\nSpreadsheet contents:\n\n{table}"
        resp = await agent.run(user_msg=prompt)  # pyright: ignore[reportDeprecated]
        return str(resp)


if __name__ == "__main__":
    print(Q19().resolve())
