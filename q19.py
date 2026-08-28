"""GAIA question #19: from the attached Excel of fast-food sales, total the sales
from food (excluding drinks), in USD with two decimals.

The sheet has one numeric column per menu item; the only beverage is "Soda", so
food = every numeric column except Soda. We compute this directly with pandas
(deterministic) instead of asking the LLM to sum, which it does inconsistently.
"""
from __future__ import annotations

import pandas as pd
from llama_index.core.llms import LLM

from gaia.base import QuestionSolver
from gaia.questions import download_attachment

DRINK_COLUMNS = {"soda"}  # the only beverage; everything else (incl. Ice Cream) is food


class Q19(QuestionSolver):
    number = 19

    async def solve(self, llm: LLM) -> str:  # llm unused: computation is deterministic
        question = self.get_question()
        df = pd.read_excel(download_attachment(question))
        numeric = df.select_dtypes("number")
        food = numeric[[c for c in numeric.columns if c.lower() not in DRINK_COLUMNS]]
        total = float(food.to_numpy().sum())
        return f"FINAL ANSWER: {total:.2f}"


if __name__ == "__main__":
    print(Q19().resolve())
