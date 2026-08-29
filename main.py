"""Registry of all question solvers. Run this file to solve them all in order."""
from __future__ import annotations


from q6 import Q6
from q7 import Q7
from q10 import Q10
from q14 import Q14
from q19 import Q19

from gaia.base import advanced_open_router_llm, reasoning_open_router_llm
from gaia.web_solver import WebSolver
from gaia.reasoning_solver import ReasoningSolver
from gaia.coding_solver import CodingSolver

# Instantiate each solver; key the registry by its GAIA question number.
# Model tiers: default gpt-4o-mini for the easy/self-contained ones; DeepSeek V3
# (advanced_open_router_llm) for #10's audio formatting; gpt-5-mini
# (reasoning_open_router_llm) for the web-retrieval and reasoning questions, where it
# is far more reliable than DeepSeek (which loops or gives up run-to-run).
SOLVERS = {
    s.number: s
    for s in [
        WebSolver(
            1,
            hint="Count only studio albums; exclude live albums, compilations, and EPs.",
        ).with_llm(reasoning_open_router_llm),
        ReasoningSolver(3),
        WebSolver(5).with_llm(reasoning_open_router_llm),
        Q6(),
        Q7(),
        WebSolver(8).with_llm(reasoning_open_router_llm),
        ReasoningSolver(
            9,
            hint=(
                "A botanical vegetable is a root, stem, leaf, or flower. Exclude anything "
                "that is botanically a fruit (develops from a flower and contains seeds), "
                "such as green beans, corn, zucchini, bell peppers, and tomatoes. "
                "Keep each item's exact wording from the list (e.g. 'fresh basil', not "
                "'basil'). Output a comma-separated list, alphabetized (e.g. 'a, b, c')."
            ),
        ).with_llm(reasoning_open_router_llm),
        Q10().with_llm(advanced_open_router_llm),
        WebSolver(11).with_llm(reasoning_open_router_llm),
        CodingSolver(12),
        WebSolver(13).with_llm(reasoning_open_router_llm),
        Q14(),
        WebSolver(15).with_llm(reasoning_open_router_llm),
        WebSolver(
            16,
            hint="Write the city name in full — use 'Saint', never the abbreviation 'St.'.",
        ),
        WebSolver(17).with_llm(reasoning_open_router_llm),
        WebSolver(18).with_llm(reasoning_open_router_llm),
        Q19(),
        WebSolver(20).with_llm(reasoning_open_router_llm),
    ]
}


if __name__ == "__main__":
    for number in sorted(SOLVERS):
        print(f"=== #{number} ===")
        print(SOLVERS[number].resolve())
