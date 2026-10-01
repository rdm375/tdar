# TDAR benchmarks

This suite compares **sample placement**, not surrogate-model sophistication. Every method is evaluated with the same oracle, fixed held-out design, and SciPy Delaunay piecewise-linear interpolant.

Methods:

- Random sampling — repeated over independent seeds.
- Latin hypercube sampling — repeated over the same seeds.
- Sobol sampling — deterministic unscrambled sequence.
- TDAR — deterministic frozen method-v1 path.

Every non-adaptive design includes the four unit-square corners. This gives all interpolants the same convex hull and prevents extrapolation behavior from contaminating the comparison. TDAR uses its standard 16-point farthest-point initializer, which also includes those corners.

The canonical problems are `quadratic`, `curved_ridge`, and `multiscale`. Default budgets are 16, 24, 32, 48, 64, 96, 128, 192, and 256 evaluations. Random and LHS use 20 trials. Validation uses a fixed 32,768-point scrambled Sobol design.

Run:

```bash
python benchmarks/run_benchmarks.py
python benchmarks/plot_results.py
```

Outputs are deliberately plain and inspectable:

- `results/trials.csv` — every individual run.
- `results/summary.csv` — means, standard deviations, medians, and 10–90% intervals.
- `results/metadata.json` — benchmark configuration.
- `figures/*-rmse.png` — error versus expensive-function evaluations.

For a quick smoke test:

```bash
python benchmarks/run_benchmarks.py --problems curved_ridge --budgets 16 32 --trials 2 --heldout 2048
```

Benchmark dependencies are development/documentation dependencies and are not imported by the `tdar` runtime package.
