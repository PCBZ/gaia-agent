"""Abstract base for per-question solvers.

Each GAIA question is solved by a QuestionSolver subclass that implements the
async `solve(llm)` method. The shared `resolve(llm=None)` entry point defaults
the LLM to GoogleGenAI (Gemini) and runs the async work synchronously.
"""
from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env once here so every solver has the API keys available.
load_dotenv(Path(__file__).parent.parent / ".env")

from llama_index.core.llms import LLM
from llama_index.llms.google_genai import GoogleGenAI

DEFAULT_MODEL = "gemini-3.6-flash"


def default_llm() -> LLM:
    """The LLM used when a solver's resolve() is called without one."""
    return GoogleGenAI(
        model=DEFAULT_MODEL,
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )


class QuestionSolver(ABC):
    """Solves one GAIA question."""

    number: int  # GAIA question number (1-based), used by the registry
    index: int   # 0-based index into load_questions()

    def resolve(self, llm: Optional[LLM] = None) -> str:
        """Public entry point. Injects GoogleGenAI unless another LLM is given."""
        return asyncio.run(self._resolve(llm))

    async def _resolve(self, llm: Optional[LLM]) -> str:
        return await self.solve(llm or default_llm())

    @abstractmethod
    async def solve(self, llm: LLM) -> str:
        """Produce the answer for this question using the given LLM."""
        raise NotImplementedError
