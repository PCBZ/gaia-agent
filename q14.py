"""GAIA question #14: listen to the professor's recording and report the page
numbers to study, as an ascending comma-delimited list.
"""
from __future__ import annotations

from llama_index.core.llms import LLM

from gaia.base import SYSTEM_PROMPT, QuestionSolver, openrouter_llm
from gaia.media import transcribe_audio
from gaia.questions import download_attachment


class Q14(QuestionSolver):
    number = 14

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        audio = download_attachment(question)
        transcript = transcribe_audio(str(audio))
        prompt = (
            f"{SYSTEM_PROMPT}\n\n{question.question}\n\n"
            "Report page numbers only, not problem numbers.\n\n"
            f"Transcript of the recording:\n{transcript}"
        )
        return str(await llm.acomplete(prompt))


if __name__ == "__main__":
    print(Q14().resolve(llm=openrouter_llm()))
