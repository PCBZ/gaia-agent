"""A generic solver for questions that ship code to run.

Downloads the attached source file and gives the model a code_interpreter tool to
execute it, then reads off the result.
"""
from __future__ import annotations

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec

from gaia.base import QuestionSolver
from gaia.questions import download_attachment

SYSTEM = (
    "You are a general AI assistant. Use the code_interpreter tool to run the "
    "code and observe its output. Then finish with a line:\nFINAL ANSWER: <number>"
)


class CodingSolver(QuestionSolver):
    """Answers a question whose attachment is code to execute. Pass the number."""

    def __init__(self, number: int) -> None:
        self.number = number

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        code = download_attachment(question).read_text(encoding="utf-8")
        agent = FunctionAgent(
            tools=CodeInterpreterToolSpec().to_tool_list(),
            llm=llm,
            system_prompt=SYSTEM,
        )
        prompt = f"{question.question}\n\n```python\n{code}\n```"
        resp = await agent.run(  # pyright: ignore[reportDeprecated]
            user_msg=prompt, max_iterations=30, early_stopping_method="generate"
        )
        return str(resp)
