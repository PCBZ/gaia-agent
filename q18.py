"""GAIA question #18: the Nippon-Ham Fighters pitchers with the number just before
and just after Taishō Tamai's (as of July 2023), last names only, Roman characters,
formatted "Before, After". Pure web retrieval — uses the generic WebSolver.

Expected answer: Yamasaki, Uehara (Tamai #19; #18 Yamasaki, #20 Uehara).
"""
from __future__ import annotations

from gaia.base import openrouter_llm
from gaia.web_solver import WebSolver

if __name__ == "__main__":
    print(WebSolver(18).resolve(llm=openrouter_llm()))
