"""Generate benchmark figures from benchmarks/results/summary.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]


def load(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["budget"] = int(r["budget"])
        for k in list(r):
            if k not in {"problem", "method", "budget", "n_trials"}:
                r[k] = float(r[k])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT / "benchmarks" / "results" / "summary.csv")
    p.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "figures")
    p.add_argument("--metric", choices=("rmse", "mae", "max_abs"), default="rmse")
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = load(args.input)

    for problem in sorted({r["problem"] for r in rows}):
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        for method in ("Random", "LHS", "Sobol", "TDAR"):
            g = sorted((r for r in rows if r["problem"] == problem and r["method"] == method), key=lambda r: r["budget"])
            if not g: continue
            x = np.array([r["budget"] for r in g]); y = np.array([r[f"{args.metric}_mean"] for r in g])
            ax.plot(x, y, marker="o", label=method)
            if method in {"Random", "LHS"}:
                lo = np.array([r[f"{args.metric}_q10"] for r in g]); hi = np.array([r[f"{args.metric}_q90"] for r in g])
                ax.fill_between(x, lo, hi, alpha=0.15)
        ax.set(xlabel="Function evaluations", ylabel=args.metric.upper(), title=problem.replace("_", " ").title())
        ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.25); ax.legend(); fig.tight_layout()
        out = args.output / f"{problem}-{args.metric}.png"; fig.savefig(out, dpi=160); plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
