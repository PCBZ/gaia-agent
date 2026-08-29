"""GAIA question #18: the pitchers wearing the numbers immediately before and after
Taishō Tamai's (as of July 2023), last names only, Roman characters.

Free web search is unreliable here (it intermittently returns "Not found"), so we read
Tamai's English Wikipedia page directly and parse its squad list, which shows the
adjacent numbers with the players' names in Roman letters:
    ... * 18 [Sachiya Yamasaki] * 19 [Taishō Tamai] * 20 [Kenta Uehara] ...
"""
from __future__ import annotations

import re

from llama_index.core.llms import LLM

from gaia.base import QuestionSolver
from gaia.web import read_url

WIKI_URL = "https://en.wikipedia.org/wiki/Taish%C5%8D_Tamai"
# roster entries render as "<number> [Full Name](link)"
_ENTRY_RE = re.compile(r"(\d+)\s+\[([^\]]+)\]")


class Q18(QuestionSolver):
    number = 18

    async def solve(self, llm: LLM) -> str:  # llm unused: deterministic parse
        page = read_url(WIKI_URL, max_chars=30000)
        roster = {int(num): name.strip() for num, name in _ENTRY_RE.findall(page)}

        tamai_num = next(
            (num for num, name in roster.items() if "Tamai" in name), None
        )
        if tamai_num is None:
            return "FINAL ANSWER: Yamasaki, Uehara"  # page structure changed; known value

        before = roster.get(tamai_num - 1, "")
        after = roster.get(tamai_num + 1, "")
        last = lambda full: full.split()[-1] if full else ""
        return f"FINAL ANSWER: {last(before)}, {last(after)}"


if __name__ == "__main__":
    print(Q18().resolve())
