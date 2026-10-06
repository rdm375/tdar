#!/usr/bin/env python3
"""Sweep TDAR budget to measure methane-Z accuracy/complexity/speed tradeoffs.

Stage 1 builds TDAR CPWA checkpoints and evaluates every checkpoint on one
frozen Sobol validation design.  If --cpwa-relu is supplied, Stage 2 compiles
Pareto candidates to exact min/max DAGs, applies bottom-up synthesis and
finite-library CP-SAT cleanup, then invokes cpwa-relu's same-device PR-vs-CPWA
benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import qmc

from tdar import TDARConfig, farthest_point, sample, save_simplicial_cpwa
from tdar.adapters.thermogpu import ThermoGPUDomain, ThermoGPUOracle


def metrics(truth, pred):
    e = np.abs(np.asarray(pred) - np.asarray(truth))
    return {
        "mae": float(np.mean(e)),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "p95_abs": float(np.quantile(e, .95)),
        "p99_abs": float(np.quantile(e, .99)),
        "max_abs": float(np.max(e)),
    }


def pareto(rows):
    """Accuracy/size nondominated rows: minimize max_abs and simplices."""
    out = []
    for r in rows:
        dominated = any(
            (q["max_abs"] <= r["max_abs"] and q["simplices"] <= r["simplices"])
            and (q["max_abs"] < r["max_abs"] or q["simplices"] < r["simplices"])
            for q in rows
        )
        if not dominated:
            out.append(r)
    return sorted(out, key=lambda r: r["budget"])


def run(cmd, *, cwd, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, env=env, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", required=True, help="path to thermogpu_tdar_oracle")
    p.add_argument("--budgets", nargs="+", type=int, default=[16,24,32,48,64,96,128])
    p.add_argument("--initial", type=int, default=16)
    p.add_argument("--fill-candidates", type=int, default=4096)
    p.add_argument("--seed", type=int, default=20261002)
    p.add_argument("--validation-points", type=int, default=10000)
    p.add_argument("--output", type=Path, default=Path("benchmarks/results-thermogpu-tradeoff"))
    p.add_argument("--cpwa-relu", type=Path, help="optional cpwa-relu checkout for synthesis/timing stage")
    p.add_argument("--batches", nargs="+", type=int, default=[1,1000,100000])
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--cpsat-time-limit", type=float, default=60.0)
    p.add_argument("--cpsat-workers", type=int, default=8)
    args = p.parse_args()

    budgets = sorted(set(args.budgets))
    if not budgets or budgets[0] < args.initial:
        p.error("all budgets must be >= --initial")
    # With the frozen batch-size-8 method, these budgets are true prefixes of
    # one refinement trajectory.  Reject partial batches rather than silently
    # changing the last refinement decision.
    if any((b - args.initial) % 8 for b in budgets):
        p.error("budgets must differ from --initial by multiples of 8 for nested checkpoints")

    out = args.output.resolve()
    artifacts = out / "artifacts"
    out.mkdir(parents=True, exist_ok=True); artifacts.mkdir(exist_ok=True)
    domain = ThermoGPUDomain()
    initial = farthest_point(args.initial, candidates=max(128, 8*args.initial), seed=args.seed)
    config = TDARConfig(fill_candidates=args.fill_candidates, seed=args.seed, record_history=False)

    # One frozen validation design and one oracle truth vector for every model.
    m = int(np.ceil(np.log2(args.validation_points)))
    validation = qmc.Sobol(d=2, scramble=True, seed=args.seed + 1).random_base2(m)[:args.validation_points]
    np.save(out / "validation-points.npy", validation)
    print(f"evaluating frozen validation truth: {len(validation)} points", flush=True)
    with ThermoGPUOracle(args.oracle, domain=domain, require_root=None) as oracle:
        truth = np.asarray([oracle(x)[0] for x in validation])
    np.save(out / "validation-truth.npy", truth)

    rows = []
    for budget in budgets:
        print(f"\n=== TDAR budget {budget} ===", flush=True)
        t0 = time.perf_counter()
        with ThermoGPUOracle(args.oracle, domain=domain) as oracle:
            result = sample(oracle, initial, budget, config)
        build_s = time.perf_counter() - t0
        path = artifacts / f"methane-z-b{budget}.npz"
        save_simplicial_cpwa(result, path, metadata={
            "target":"methane-Z", "budget":budget, "initial_points":args.initial,
            "seed":args.seed, "temperature_K":list(domain.temperature_K),
            "pressure_Pa":list(domain.pressure_Pa), "validation_seed":args.seed+1,
        })
        pred = np.asarray(LinearNDInterpolator(result.points, result.values)(validation), float)
        if np.isnan(pred).any():
            raise RuntimeError("validation points fell outside TDAR convex hull")
        r = {"budget":budget, "points":len(result.points), "simplices":len(result.simplices),
             "tdar_seconds":build_s, "artifact":str(path)}
        r.update(metrics(truth, pred)); rows.append(r)
        print("simplices={simplices} rmse={rmse:.6g} p99={p99_abs:.6g} max={max_abs:.6g}".format(**r))

    front = pareto(rows)
    front_budgets = {r["budget"] for r in front}
    for r in rows: r["pareto_accuracy_complexity"] = r["budget"] in front_budgets

    if args.cpwa_relu:
        cpwa = args.cpwa_relu.resolve()
        env = os.environ.copy()
        # Prefer checkout sources over any installed cpwa-relu.
        env["PYTHONPATH"] = str(cpwa / "src") + os.pathsep + env.get("PYTHONPATH", "")
        for r in front:
            b = r["budget"]
            print(f"\n=== synthesize/benchmark Pareto budget {b} ===", flush=True)
            native = artifacts / f"methane-z-b{b}-native.cpwa"
            bottom = artifacts / f"methane-z-b{b}-bottom-up.cpwa"
            final = artifacts / f"methane-z-b{b}-cpsat.cpwa"
            run([sys.executable, "benchmarks/build_native_dag_artifact.py", "--source", r["artifact"], "--output", native], cwd=cpwa, env=env)
            run([sys.executable, "benchmarks/synthesize_bottom_up.py", "--artifact", native, "--check-points", "0", "--output", bottom], cwd=cpwa, env=env)
            run([sys.executable, "benchmarks/optimize_min_circuit_cpsat.py", "--artifact", bottom, "--intersections", "0", "--time-limit", args.cpsat_time_limit, "--workers", args.cpsat_workers, "--output", final], cwd=cpwa, env=env)
            timing_dir = out / f"timing-b{b}"
            run([sys.executable, "benchmarks/benchmark_surrogate_crossover.py", "--artifact", final,
                 "--dtypes", "float32", "--batches", *map(str,args.batches), "--repeats", str(args.repeats),
                 "--warmup", str(args.warmup), "--seed", str(args.seed+2), "--output", timing_dir], cwd=cpwa, env=env)
            with (timing_dir / "summary.csv").open() as f:
                timing = list(csv.DictReader(f))
            r["timings"] = {x["batch"]: {
                "direct_ns_per_eval": float(x["direct_ns_per_eval"]),
                "surrogate_ns_per_eval": float(x["surrogate_ns_per_eval"]),
                "direct_over_surrogate_speedup": float(x["direct_over_surrogate_speedup"]),
            } for x in timing if x["dtype"] == "float32"}
            # Read final DAG counts without coupling this driver to cpwa internals.
            code = ("from cpwa_relu import load_native_dag; import json,sys; d=load_native_dag(sys.argv[1]); "
                    "print(json.dumps({'affine':d.affine_nodes,'min':d.min_nodes,'max':d.max_nodes,'nodes':len(d.nodes)}))")
            q = subprocess.run([sys.executable,"-c",code,str(final)], cwd=cpwa, env=env, check=True, text=True, capture_output=True)
            r.update({f"dag_{k}":v for k,v in json.loads(q.stdout).items()})
            r["final_artifact"] = str(final)

    # Flatten CSV while retaining full nested timing data in JSON.
    flat = []
    for r in rows:
        z = {k:v for k,v in r.items() if k != "timings"}
        for batch, t in r.get("timings", {}).items():
            for k,v in t.items(): z[f"batch_{batch}_{k}"] = v
        flat.append(z)
    keys=[]
    for r in flat:
        for k in r:
            if k not in keys: keys.append(k)
    with (out/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(flat)
    report={"domain":{"temperature_K":domain.temperature_K,"pressure_Pa":domain.pressure_Pa},
            "budgets":budgets,"validation_points":args.validation_points,"seed":args.seed,
            "pareto_budgets":[r["budget"] for r in front],"results":rows}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("\n=== accuracy/complexity Pareto budgets ===")
    print(" ".join(map(str, report["pareto_budgets"])))
    print(f"wrote: {out/'summary.csv'}")
    print(f"wrote: {out/'report.json'}")

if __name__ == "__main__": main()
