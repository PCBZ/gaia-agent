"""Abstract base for per-question solvers.

Each GAIA question is solved by a QuestionSolver subclass that implements the
async `solve(llm)` method. The shared `resolve(llm=None)` entry point picks the
LLM as: the given llm, else the solver's own `.llm`, else `default_llm()`
(OpenRouter gpt-4o-mini), and runs the async work synchronously.
"""
from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

# Load .env once here so every solver has the API keys available.
load_dotenv(Path(__file__).parent.parent / ".env")

from llama_index.core.llms import LLM
from llama_index.llms.google_genai import GoogleGenAI

from gaia.config import CONFIG

DEFAULT_MODEL = CONFIG["llm"]["default_model"]

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
    "- Open the actual URLs returned by web_search with read_url; never invent or "
    "guess a URL.\n"
    "- Base the answer only on text you actually retrieved. Never fabricate a name, "
    "number, or fact to fill a gap — if you have not found it, keep searching and "
    "reading. Do not fall back on an author's or a nearby name as the answer.\n"
    "- When the question names a specific source (a site, document, page, or an "
    "author's materials), find and read THAT exact source; do not trust third-party "
    "answer, quiz, or crossword-aggregator sites.\n"
    "- Give names and list items exactly as the source states them, with all "
    "descriptive words (e.g. 'freshly squeezed lemon juice', not 'lemon juice'; a "
    "complete city name). Do not abbreviate, shorten, or drop any part.\n"
    "Finish with a single line:\n"
    "FINAL ANSWER: <answer>\n"
    "After the colon put only the answer, formatted exactly as asked: a number "
    "with no commas or units unless requested; otherwise as few words as possible, "
    "or a comma-separated list. No extra words or trailing punctuation."
)


def gemini_llm() -> LLM:
    """Gemini via AI Studio (needs GEMINI_API_KEY + credit). Not the default: its
    free tier is tiny and prepay may be exhausted."""
    return GoogleGenAI(
        model=DEFAULT_MODEL,
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
        max_retries=8,  # ride out the free-tier per-minute limit (5/min, 429)
    )


def default_llm() -> LLM:
    """The LLM used when a solver's resolve() is called without one — OpenRouter
    gpt-4o-mini (reliable and cheap; the Gemini/HF free tiers are exhausted)."""
    return openrouter_llm()


def huggingface_llm(model: str = CONFIG["llm"]["huggingface"]["model"]) -> LLM:
    """An HF Inference Providers LLM (OpenAI-compatible router), using HF_TOKEN.

    A separate free-tier quota from Gemini. Inject it explicitly, e.g.
    `Q17().resolve(llm=huggingface_llm())`. Pick a model that supports tool
    calling, since FunctionAgent needs it.
    """
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=model,
        api_base=CONFIG["llm"]["huggingface"]["api_base"],
        api_key=os.environ["HF_TOKEN"],
        is_chat_model=True,
        is_function_calling_model=True,
        temperature=0,
    )


def groq_llm(model: str = CONFIG["llm"]["groq"]["model"]) -> LLM:
    """A Groq-hosted LLM (OpenAI-compatible), using GROQ_API_KEY.

    Groq's free tier is generous and needs no payment info. Inject explicitly,
    e.g. `Q15().resolve(llm=groq_llm())`. Pick a tool-calling model.
    """
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=model,
        api_base=CONFIG["llm"]["groq"]["api_base"],
        api_key=os.environ["GROQ_API_KEY"],
        is_chat_model=True,
        is_function_calling_model=True,
        temperature=0,
    )


def openrouter_llm(model: str = CONFIG["llm"]["openrouter"]["model"]) -> LLM:
    """An OpenRouter-hosted LLM (OpenAI-compatible), using OPENROUTER_API_KEY.

    OpenRouter's ':free' models cost no tokens (rate-limited). Pick one that
    supports tool calling, since FunctionAgent needs it. Inject explicitly,
    e.g. `Q15().resolve(llm=openrouter_llm())`.
    """
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=model,
        api_base=CONFIG["llm"]["openrouter"]["api_base"],
        api_key=os.environ["OPENROUTER_API_KEY"],
        is_chat_model=True,
        is_function_calling_model=True,
        temperature=0,
        max_retries=8,  # ride out free-tier per-minute rate limits (429 + Retry-After)
        timeout=120,
    )


def advanced_open_router_llm() -> LLM:
    """A stronger OpenRouter model (DeepSeek V3 by default) for questions the cheap
    default gets wrong — ~10x cheaper than gpt-4o with reliable tool calling."""
    return openrouter_llm(CONFIG["llm"]["openrouter"]["advanced_model"])


class QuestionSolver(ABC):
    """Solves one GAIA question."""

    number: int  # GAIA question number (1-based), used by the registry
    # A factory called lazily at solve time to build this solver's LLM (so importing
    # the registry never touches API keys). None -> use default_llm().
    llm_factory: Optional[Callable[[], LLM]] = None

    @property
    def index(self) -> int:
        """0-based index into load_questions(); derived from `number`."""
        return self.number - 1

    def get_question(self):
        """Return the Question object for this solver."""
        from gaia.questions import load_questions

        return load_questions()[self.index]

    def with_llm(self, factory: Callable[[], LLM]) -> "QuestionSolver":
        """Set a factory (called lazily at solve time) for this solver's LLM.

        Pass a zero-arg callable, e.g. `.with_llm(lambda: openrouter_llm("openai/gpt-4o"))`,
        so importing the registry never constructs an LLM or reads API keys.
        """
        self.llm_factory = factory
        return self

    def resolve(self, llm: Optional[LLM] = None) -> str:
        """Public entry point. Uses the given llm, else this solver's factory, else default."""
        return asyncio.run(self._resolve(llm))

    async def _resolve(self, llm: Optional[LLM]) -> str:
        chosen = llm or (self.llm_factory() if self.llm_factory else default_llm())
        return await self.solve(chosen)

    @abstractmethod
    async def solve(self, llm: LLM) -> str:
        """Produce the answer for this question using the given LLM."""
        raise NotImplementedError
