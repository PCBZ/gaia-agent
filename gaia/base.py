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

DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Generic, question-agnostic policy prompt shared by every solver. It encodes
# problem-solving discipline (not facts about any specific question) plus GAIA's
# exact-match answer format.
SYSTEM_PROMPT = (
    "You are a general AI assistant. Use the available tools to gather evidence; "
    "do not rely on memory or assumptions for facts.\n"
    "- Verify before concluding. When the question asks for the most / least / "
    "largest / smallest / first / only / best, collect the complete set of "
    "candidates and confirm none beats your pick — never answer from the first "
    "plausible candidate you find.\n"
    "- A page's key tables or lists may appear far down; read enough to be sure.\n"
    "- To find a maximum/minimum/ranking over many rows, load the data and compute "
    "it with the code_interpreter tool rather than comparing by eye.\n"
    "- Recompute rather than estimate when numbers are involved.\n"
    "Finish with a single line:\n"
    "FINAL ANSWER: <answer>\n"
    "After the colon put only the answer, formatted exactly as asked: a number "
    "with no commas or units unless requested; otherwise as few words as possible, "
    "or a comma-separated list. No extra words or trailing punctuation."
)


def default_llm() -> LLM:
    """The LLM used when a solver's resolve() is called without one."""
    return GoogleGenAI(
        model=DEFAULT_MODEL,
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
        max_retries=8,  # ride out the free-tier per-minute limit (5/min, 429)
    )


def huggingface_llm(model: str = "Qwen/Qwen2.5-72B-Instruct") -> LLM:
    """An HF Inference Providers LLM (OpenAI-compatible router), using HF_TOKEN.

    A separate free-tier quota from Gemini. Inject it explicitly, e.g.
    `Q17().resolve(llm=huggingface_llm())`. Pick a model that supports tool
    calling, since FunctionAgent needs it.
    """
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=model,
        api_base="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
        is_chat_model=True,
        is_function_calling_model=True,
        temperature=0,
    )


def groq_llm(model: str = "openai/gpt-oss-120b") -> LLM:
    """A Groq-hosted LLM (OpenAI-compatible), using GROQ_API_KEY.

    Groq's free tier is generous and needs no payment info. Inject explicitly,
    e.g. `Q15().resolve(llm=groq_llm())`. Pick a tool-calling model.
    """
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=model,
        api_base="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        is_chat_model=True,
        is_function_calling_model=True,
        temperature=0,
    )


def openrouter_llm(model: str = "openai/gpt-4o-mini") -> LLM:
    """An OpenRouter-hosted LLM (OpenAI-compatible), using OPENROUTER_API_KEY.

    OpenRouter's ':free' models cost no tokens (rate-limited). Pick one that
    supports tool calling, since FunctionAgent needs it. Inject explicitly,
    e.g. `Q15().resolve(llm=openrouter_llm())`.
    """
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=model,
        api_base="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        is_chat_model=True,
        is_function_calling_model=True,
        temperature=0,
        max_retries=8,  # ride out free-tier per-minute rate limits (429 + Retry-After)
        timeout=120,
    )


class QuestionSolver(ABC):
    """Solves one GAIA question."""

    number: int  # GAIA question number (1-based), used by the registry

    @property
    def index(self) -> int:
        """0-based index into load_questions(); derived from `number`."""
        return self.number - 1

    def get_question(self):
        """Return the Question object for this solver."""
        from gaia.questions import load_questions

        return load_questions()[self.index]

    def resolve(self, llm: Optional[LLM] = None) -> str:
        """Public entry point. Injects GoogleGenAI unless another LLM is given."""
        return asyncio.run(self._resolve(llm))

    async def _resolve(self, llm: Optional[LLM]) -> str:
        return await self.solve(llm or default_llm())

    @abstractmethod
    async def solve(self, llm: LLM) -> str:
        """Produce the answer for this question using the given LLM."""
        raise NotImplementedError
