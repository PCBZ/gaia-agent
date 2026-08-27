"""Entry point: dispatch to a question's solver via a registry."""
from __future__ import annotations

import argparse

from q1 import Q1
from q3 import Q3
from q5 import Q5
from q6 import Q6
from q7 import Q7
from q9 import Q9
from q10 import Q10
from q11 import Q11
from q12 import Q12
from q13 import Q13
from q14 import Q14
from q15 import Q15
from q17 import Q17
from q19 import Q19
from q20 import Q20

from gaia.web_solver import WebSolver

# Instantiate each solver; key the registry by its GAIA question number.
SOLVERS = {s.number: s for s in [Q1(), Q3(), Q5(), Q6(), Q7(), Q9(), Q10(), Q11(), Q12(), Q13(), Q14(), Q15(), Q17(), Q19(), Q20(), WebSolver(8), WebSolver(16), WebSolver(18)]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a GAIA question solver")
    parser.add_argument(
        "-n", "--number", type=int, default=3,
        help=f"question number to solve (available: {sorted(SOLVERS)})",
    )
    args = parser.parse_args()

    solver = SOLVERS.get(args.number)
    if solver is None:
        parser.error(f"no solver for #{args.number}; available: {sorted(SOLVERS)}")
    print(solver.resolve())


if __name__ == "__main__":
    main()
