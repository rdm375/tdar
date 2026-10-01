"""Reproducible Random/LHS/Sobol/TDAR benchmark suite.

The benchmark isolates sample placement: every method uses the same scalar
oracle, the same held-out points, and the same piecewise-linear surrogate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import jax

# Benchmark functions are evaluated in double precision for reproducibility.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from problems import curved_ridge, multiscale, quadratic
from tdar import TDARConfig, farthest_point, sample, jax_oracle
from baselines import lhs_design, random_design, sobol_design
from metrics import error_metrics, piecewise_linear_predict

PROBLEMS = {
    "quadratic": quadratic,
    "curved_ridge": curved_ridge,
    "multiscale": multiscale,
}
DEFAULT_BUDGETS = (16, 24, 32, 48, 64, 96, 128, 192, 256)


def evaluate(f, x: np.ndarray) -> np.ndarray:
    return np.asarray(jax.vmap(f)(jnp.asarray(x)), dtype=float)


def heldout_design(n: int) -> np.ndarray:
    """Fixed deterministic Sobol held-out set shared by every method/trial."""
    # Scrambled with a fixed seed avoids placing validation points exactly on
    # the benchmark design while remaining completely reproducible.
    return qmc.Sobol(d=2, scramble=True, seed=20261001).random(n)


def benchmark(problem_names, budgets, trials, heldout_n):
    test_x = heldout_design(heldout_n)
    rows = []

    for problem_name in problem_names:
        f = PROBLEMS[problem_name]
        truth = evaluate(f, test_x)
        initial = farthest_point(16)

        for budget in budgets:
            # TDAR is deterministic for fixed configuration and initializer.
            tdar_result = sample(jax_oracle(f), initial, budget=budget, config=TDARConfig())
            pred = piecewise_linear_predict(tdar_result.x, tdar_result.y, test_x)
            rows.append({"problem": problem_name, "method": "TDAR", "budget": budget,
                         "trial": 0, "seed": TDARConfig().seed, **error_metrics(truth, pred)})

            x = sobol_design(budget)
            y = evaluate(f, x)
            pred = piecewise_linear_predict(x, y, test_x)
            rows.append({"problem": problem_name, "method": "Sobol", "budget": budget,
                         "trial": 0, "seed": -1, **error_metrics(truth, pred)})

            for trial in range(trials):
                seed = 1000 + trial
                for method, design in (("Random", random_design(budget, seed)),
                                       ("LHS", lhs_design(budget, seed))):
                    y = evaluate(f, design)
                    pred = piecewise_linear_predict(design, y, test_x)
                    rows.append({"problem": problem_name, "method": method, "budget": budget,
                                 "trial": trial, "seed": seed, **error_metrics(truth, pred)})
    return rows


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row["problem"], row["method"], row["budget"])
        grouped.setdefault(key, []).append(row)
    out = []
    for (problem, method, budget), group in sorted(grouped.items()):
        record = {"problem": problem, "method": method, "budget": budget, "n_trials": len(group)}
        for metric in ("rmse", "mae", "max_abs"):
            a = np.asarray([r[metric] for r in group])
            record[f"{metric}_mean"] = float(a.mean())
            record[f"{metric}_std"] = float(a.std(ddof=1)) if len(a) > 1 else 0.0
            record[f"{metric}_median"] = float(np.median(a))
            record[f"{metric}_q10"] = float(np.quantile(a, 0.10))
            record[f"{metric}_q90"] = float(np.quantile(a, 0.90))
        out.append(record)
    return out


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problems", nargs="+", choices=PROBLEMS, default=list(PROBLEMS))
    p.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--heldout", type=int, default=32768)
    p.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "results")
    args = p.parse_args()
    if min(args.budgets) < 16:
        p.error("budgets must be >= 16 because TDAR uses a 16-point initializer")

    rows = benchmark(args.problems, args.budgets, args.trials, args.heldout)
    summary = summarize(rows)
    write_csv(args.output / "trials.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    metadata = {
        "problems": args.problems, "budgets": args.budgets, "random_lhs_trials": args.trials,
        "heldout_points": args.heldout, "heldout_generator": "scrambled Sobol seed=20261001",
        "surrogate": "scipy.interpolate.LinearNDInterpolator",
        "common_domain_points": "four unit-square corners included in every baseline design",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {len(rows)} trials and {len(summary)} summary rows to {args.output}")


if __name__ == "__main__":
    main()
