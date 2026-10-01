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

## Oracle-call cost benchmark

`run_cost_benchmarks.py` is a separate experiment that compares methods under a
budget of scalar function calls. Random, LHS, and Sobol spend one call per
location. TDAR obtains gradients by bounded finite differences and is charged
for every perturbation: 3 calls/location for forward differences and 5 for
central differences in 2-D. Near the unit-square boundary the finite-difference
oracle switches to inward one-sided formulas and never samples outside the
domain.

The default study evaluates forward and central differences at steps `1e-3`,
`1e-4`, `1e-5`, and `1e-6` using call budgets 48 through 768. Exact/autodiff
TDAR is deliberately not assigned an artificial equivalent call cost; the
existing `run_benchmarks.py` remains the sample-location-efficiency experiment.

```bash
python benchmarks/run_cost_benchmarks.py
```

For a quick smoke test:

```bash
python benchmarks/run_cost_benchmarks.py \
  --problems curved_ridge --call-budgets 48 96 --trials 2 --heldout 2048 \
  --schemes forward central --steps 1e-4
```


## Canonical oracle-cost benchmark

The canonical cost-normalized experiment uses `h = 1e-3` for both forward and
central finite differences. This value was selected after a separate float64
sensitivity sweep on the curved-ridge problem; it is not tuned independently
for each problem or call budget. The default call-budget grid is
`48, 64, 96, 128, 160, 192, 256, 384, 512, 768`.

Run the canonical experiment with:

```bash
python benchmarks/run_cost_benchmarks.py
```

The broad finite-difference step-size study remains reproducible as a separate
experiment:

```bash
python benchmarks/run_fd_sensitivity.py --problems curved_ridge \
  --call-budgets 48 96 192 384 --trials 5 --heldout 8192
```

This separation prevents step-size sweeps from being confused with the
canonical method comparison.
