"""Registry of all question solvers. Run this file to solve them all in order."""
from __future__ import annotations


from q7 import Q7
from q10 import Q10
from q14 import Q14
from q19 import Q19

from gaia.base import advanced_open_router_llm, premium_open_router_llm
from gaia.web_solver import WebSolver
from gaia.reasoning_solver import ReasoningSolver
from gaia.coding_solver import CodingSolver

# Instantiate each solver; key the registry by its GAIA question number.
# Model tiers (cheapest first): default gpt-4o-mini -> advanced DeepSeek V3
# (.with_llm(advanced_open_router_llm), for questions mini failed) -> premium gpt-4o
# (.with_llm(premium_open_router_llm), for #8, which even DeepSeek can't locate).
SOLVERS = {
    s.number: s
    for s in [
        WebSolver(1).with_llm(advanced_open_router_llm),
        ReasoningSolver(3),
        WebSolver(5),
        ReasoningSolver(6).with_llm(advanced_open_router_llm),
        Q7(),
        WebSolver(8).with_llm(premium_open_router_llm),
        ReasoningSolver(9).with_llm(advanced_open_router_llm),
        Q10().with_llm(advanced_open_router_llm),
        WebSolver(11),
        CodingSolver(12),
        WebSolver(13).with_llm(advanced_open_router_llm),
        Q14(),
        WebSolver(15),
        WebSolver(16).with_llm(advanced_open_router_llm),
        WebSolver(17).with_llm(advanced_open_router_llm),
        WebSolver(18).with_llm(advanced_open_router_llm),
        Q19().with_llm(advanced_open_router_llm),
        WebSolver(20),
    ]
}


if __name__ == "__main__":
    for number in sorted(SOLVERS):
        print(f"=== #{number} ===")
        print(SOLVERS[number].resolve())
