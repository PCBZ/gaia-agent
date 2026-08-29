"""GAIA question #6: given a Cayley table for * on S = {a,b,c,d,e}, return the subset
of elements involved in any counter-example to commutativity (i.e. some x*y != y*x).

This is a pure computation, not a judgement call, so we parse the markdown table from
the question and check every pair deterministically instead of asking an LLM (which
intermittently miscounts, e.g. adding an element not actually involved).
"""
from __future__ import annotations

from llama_index.core.llms import LLM

from gaia.base import QuestionSolver


def _parse_table(question: str) -> tuple[list[str], dict[tuple[str, str], str]]:
    """Parse the markdown Cayley table into (elements, op[(row, col)] = value)."""
    rows = []
    for line in question.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-") for c in cells):  # |---|---| separator row
            continue
        rows.append(cells)
    header = rows[0]
    elements = header[1:]  # drop the "*" corner
    op = {
        (row[0], col): val
        for row in rows[1:]
        for col, val in zip(elements, row[1:])
    }
    return elements, op


class Q6(QuestionSolver):
    number = 6

    async def solve(self, llm: LLM) -> str:  # llm unused: computation is deterministic
        elements, op = _parse_table(self.get_question().question)
        involved: set[str] = set()
        for x in elements:
            for y in elements:
                if x < y and op[(x, y)] != op[(y, x)]:
                    involved.update((x, y))
        return f"FINAL ANSWER: {', '.join(sorted(involved))}"


if __name__ == "__main__":
    print(Q6().resolve())
