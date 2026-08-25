"""GAIA question #12: "What is the final numeric output from the attached Python
code?"

Downloads the .py from the GAIA dataset and gives the model LlamaIndex's official
CodeInterpreterToolSpec (subprocess execution). Safe because we inspected the file.

Requires: pip install llama-index-tools-code-interpreter
"""
from __future__ import annotations

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import LLM
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec

from gaia.base import QuestionSolver
from gaia.questions import download_attachment, load_questions

SYSTEM = (
    "You are a general AI assistant. Use the code_interpreter tool to run the "
    "code and observe its output. Then finish with a line:\nFINAL ANSWER: <number>"
)


class Q12(QuestionSolver):
    number = 12
    index = 11

    async def solve(self, llm: LLM) -> str:
        question = load_questions()[self.index]
        code = download_attachment(question).read_text(encoding="utf-8")

        agent = FunctionAgent(tools=CodeInterpreterToolSpec().to_tool_list(), 
                              llm=llm, 
                              system_prompt=SYSTEM)

        prompt = f"{question.question}\n\n```python\n{code}\n```"
        resp = await agent.run(user_msg=prompt)  # pyright: ignore[reportDeprecated]
        return str(resp)


if __name__ == "__main__":
    print(Q12().resolve())
