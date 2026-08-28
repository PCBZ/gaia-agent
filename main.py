"""Registry of all question solvers. Run this file to solve them all in order."""
from __future__ import annotations


from q7 import Q7
from q10 import Q10
from q14 import Q14
from q19 import Q19

from gaia.base import openrouter_llm
from gaia.web_solver import WebSolver
from gaia.reasoning_solver import ReasoningSolver
from gaia.coding_solver import CodingSolver

# Instantiate each solver; key the registry by its GAIA question number.
# #1, #8, #16, #18 are plain retrieval questions handled by the generic WebSolver.
SOLVERS = {
    s.number: s
    for s in [
        WebSolver(1),
        ReasoningSolver(3),
        WebSolver(5),
        ReasoningSolver(6),
        Q7(),
        # #8 wanders to distractor pages on weaker models; gpt-4o stays on source.
        WebSolver(8).with_llm(openrouter_llm("openai/gpt-4o")),
        ReasoningSolver(9),
        Q10(),
        WebSolver(11),
        CodingSolver(12),
        WebSolver(13),
        Q14(),
        WebSolver(15),
        WebSolver(16),
        WebSolver(17),
        WebSolver(18),
        Q19(),
        WebSolver(20),
    ]
}


if __name__ == "__main__":
    for number in sorted(SOLVERS):
        print(f"=== #{number} ===")
        print(SOLVERS[number].resolve())
