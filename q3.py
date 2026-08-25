"""Solve GAIA question #3 (the reversed-sentence question).

- `solve()` is the function main.py calls.
- Running this file directly (its own __main__) tests just this question.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from llama_index.llms.google_genai import GoogleGenAI

from gaia.questions import load_questions

SYSTEM = (
    "You are a general AI assistant. Solve the task, then finish with a line:\n"
    "FINAL ANSWER: <answer>\n"
)


def solve() -> str:
    """Load question #3 and return the model's answer."""
    question = load_questions()[2]  # #3, 0-based index
    llm = GoogleGenAI(
        model="gemini-3.6-flash",
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )
    resp = llm.complete(SYSTEM + "\n\n" + question.question)
    return str(resp)


if __name__ == "__main__":
    print(solve())
