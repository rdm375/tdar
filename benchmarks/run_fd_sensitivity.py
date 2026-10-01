"""Finite-difference step-size sensitivity study for TDAR.

This is intentionally separate from the canonical oracle-cost benchmark.
It sweeps h broadly to test whether TDAR refinement decisions are robust to
finite-difference gradient accuracy.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]

# Reuse the cost benchmark implementation and override only its CLI defaults.
import run_cost_benchmarks as cost


def main():
    args = sys.argv[1:]
    if "--steps" not in args:
        args += ["--steps", "1e-2", "3e-3", "1e-3", "3e-4", "1e-4", "3e-5", "1e-5", "1e-6"]
    if "--output" not in args:
        args += ["--output", str(ROOT / "benchmarks" / "results-fd-sensitivity")]
    old = sys.argv
    try:
        sys.argv = [old[0], *args]
        cost.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
