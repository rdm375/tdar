"""Oracle-cost benchmark for TDAR with finite-difference gradients.

Unlike ``run_benchmarks.py`` (sample-location efficiency), this suite charges
finite-difference TDAR for every scalar function evaluation used to construct
its gradients. Random/LHS/Sobol spend one function call per sampled location.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import jax

# Finite-difference sensitivity studies require double precision.  Make the
# benchmark independent of the user's ambient JAX precision configuration.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve(); ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "examples"))
from problems import curved_ridge, multiscale, quadratic
from tdar import TDARConfig, farthest_point, sample
from baselines import lhs_design, random_design, sobol_design
from finite_difference import finite_difference_oracle
from metrics import error_metrics, piecewise_linear_predict

PROBLEMS = {"quadratic": quadratic, "curved_ridge": curved_ridge, "multiscale": multiscale}
DEFAULT_CALL_BUDGETS = (48, 64, 96, 128, 192, 256, 384, 512, 768)


def evaluate(f, x): return np.asarray(jax.vmap(f)(jnp.asarray(x)), dtype=float)
def scalar_function(f): return lambda x: float(np.asarray(f(jnp.asarray(x))))
def heldout_design(n): return qmc.Sobol(d=2, scramble=True, seed=20261001).random(n)


def _row(problem, method, call_budget, locations, calls_used, trial, seed, truth, pred, **extra):
    return {"problem": problem, "method": method, "call_budget": call_budget,
            "sample_locations": locations, "function_calls": calls_used,
            "unused_calls": call_budget-calls_used, "trial": trial, "seed": seed,
            **extra, **error_metrics(truth, pred)}


def benchmark(problem_names, call_budgets, trials, heldout_n, schemes, steps):
    test_x = heldout_design(heldout_n); rows=[]
    initial = farthest_point(16)
    for problem_name in problem_names:
        f=PROBLEMS[problem_name]; truth=evaluate(f,test_x); scalar=scalar_function(f)
        for call_budget in call_budgets:
            # Derivative-free baselines: one scalar call per location.
            x=sobol_design(call_budget); y=evaluate(f,x)
            rows.append(_row(problem_name,"Sobol",call_budget,len(x),len(x),0,-1,truth,
                             piecewise_linear_predict(x,y,test_x),gradient_scheme="none",fd_step=""))
            for trial in range(trials):
                seed=1000+trial
                for method,design in (("Random",random_design(call_budget,seed)),("LHS",lhs_design(call_budget,seed))):
                    y=evaluate(f,design)
                    rows.append(_row(problem_name,method,call_budget,len(design),len(design),trial,seed,truth,
                                     piecewise_linear_predict(design,y,test_x),gradient_scheme="none",fd_step=""))

            # TDAR finite differences: charge the exact calls made by the oracle.
            for scheme in schemes:
                per_location=3 if scheme=="forward" else 5
                locations=call_budget//per_location
                if locations < len(initial):
                    continue
                for step in steps:
                    oracle,counter=finite_difference_oracle(scalar,scheme=scheme,step=step)
                    r=sample(oracle,initial,budget=locations,config=TDARConfig())
                    assert counter.calls == per_location*locations
                    rows.append(_row(problem_name,f"TDAR-{scheme}-FD",call_budget,locations,counter.calls,0,
                                     TDARConfig().seed,truth,piecewise_linear_predict(r.x,r.y,test_x),
                                     gradient_scheme=scheme,fd_step=step))
    return rows


def summarize(rows):
    grouped={}
    for r in rows:
        key=(r["problem"],r["method"],r["call_budget"],r["gradient_scheme"],str(r["fd_step"]))
        grouped.setdefault(key,[]).append(r)
    out=[]
    for key,group in sorted(grouped.items()):
        problem,method,budget,scheme,step=key
        rec={"problem":problem,"method":method,"call_budget":budget,"gradient_scheme":scheme,"fd_step":step,
             "n_trials":len(group),"sample_locations_mean":float(np.mean([r["sample_locations"] for r in group])),
             "function_calls_mean":float(np.mean([r["function_calls"] for r in group]))}
        for metric in ("rmse","mae","max_abs"):
            a=np.asarray([r[metric] for r in group]); rec[f"{metric}_mean"]=float(a.mean()); rec[f"{metric}_std"]=float(a.std(ddof=1)) if len(a)>1 else 0.0
            rec[f"{metric}_median"]=float(np.median(a)); rec[f"{metric}_q10"]=float(np.quantile(a,.1)); rec[f"{metric}_q90"]=float(np.quantile(a,.9))
        out.append(rec)
    return out


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--problems",nargs="+",choices=PROBLEMS,default=list(PROBLEMS))
    p.add_argument("--call-budgets",nargs="+",type=int,default=list(DEFAULT_CALL_BUDGETS))
    p.add_argument("--trials",type=int,default=20); p.add_argument("--heldout",type=int,default=32768)
    p.add_argument("--schemes",nargs="+",choices=("forward","central"),default=["forward","central"])
    p.add_argument("--steps",nargs="+",type=float,default=[1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5,1e-6])
    p.add_argument("--output",type=Path,default=ROOT/"benchmarks"/"results-cost")
    a=p.parse_args()
    if min(a.call_budgets)<48: p.error("call budgets must be >= 48 (16 initial TDAR locations × 3 forward-FD calls)")
    rows=benchmark(a.problems,a.call_budgets,a.trials,a.heldout,a.schemes,a.steps); summary=summarize(rows)
    write_csv(a.output/"trials.csv",rows); write_csv(a.output/"summary.csv",summary)
    meta={"experiment":"oracle-call efficiency","call_budgets":a.call_budgets,"random_lhs_trials":a.trials,
          "heldout_points":a.heldout,"heldout_generator":"scrambled Sobol seed=20261001",
          "surrogate":"scipy.interpolate.LinearNDInterpolator","oracle_precision":"float64","jax_enable_x64":bool(jax.config.x64_enabled),"fd_schemes":a.schemes,"fd_steps":a.steps,
          "cost_accounting":{"Random/LHS/Sobol":"1 scalar function call per location","TDAR-forward-FD":"3 scalar function calls per location","TDAR-central-FD":"5 scalar function calls per location"},
          "note":"Exact/autodiff gradients are intentionally excluded from cost-normalized rankings because their application-dependent derivative cost is unspecified."}
    (a.output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(f"wrote {len(rows)} trials and {len(summary)} summary rows to {a.output}")

if __name__=="__main__": main()
