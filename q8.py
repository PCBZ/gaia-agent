"""GAIA question #8: the surname of the equine veterinarian mentioned in the 1.E
Exercises of LibreTexts' Introductory Chemistry (Alviar-Agnew & Agnew). Pure web
retrieval — uses the generic WebSolver.

Needs jina_search (JINA_API_KEY) to surface the LibreTexts page. gpt-4o-mini is
unreliable here (wanders to distractor pages -> Agnew/Kustritz); gpt-4o stays on
the named source and reliably reads the answer.

Expected answer: Louvrier.
"""
from __future__ import annotations

from gaia.base import openrouter_llm
from gaia.web_solver import WebSolver

if __name__ == "__main__":
    print(WebSolver(8).resolve(llm=openrouter_llm("openai/gpt-4o")))
