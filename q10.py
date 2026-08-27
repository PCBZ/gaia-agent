"""GAIA question #10: listen to the attached recipe voice memo and list the pie
filling ingredients (comma-separated, alphabetized, no measurements).
"""
from __future__ import annotations

from llama_index.core.llms import LLM

from gaia.base import SYSTEM_PROMPT, QuestionSolver, openrouter_llm
from gaia.media import transcribe_audio
from gaia.questions import download_attachment


class Q10(QuestionSolver):
    number = 10

    async def solve(self, llm: LLM) -> str:
        question = self.get_question()
        audio = download_attachment(question)
        transcript = transcribe_audio(str(audio))
        prompt = (
            f"{SYSTEM_PROMPT}\n\n{question.question}\n\n"
            "List each ingredient using its exact noun phrase from the transcript, "
            "keeping every preceding descriptive or preparation word (e.g. 'freshly "
            "squeezed lemon juice', 'granulated sugar', 'ripe strawberries'). Do not "
            "simplify an item to its base ingredient.\n\n"
            f"Transcript of the recipe audio:\n{transcript}"
        )
        return str(await llm.acomplete(prompt))


if __name__ == "__main__":
    print(Q10().resolve(llm=openrouter_llm()))
