"""Entry point: dispatch to a question's solver via a registry."""
from __future__ import annotations

import argparse

from q3 import Q3
from q6 import Q6
from q12 import Q12

# Instantiate each solver; key the registry by its GAIA question number.
SOLVERS = {s.number: s for s in [Q3(), Q6(), Q12()]}


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
