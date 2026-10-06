"""Build a TDAR methane-Z surrogate using ThermoGPU's persistent oracle."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tdar import TDARConfig, farthest_point, sample, save_simplicial_cpwa
from tdar.adapters.thermogpu import ThermoGPUDomain, ThermoGPUOracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, help="path to thermogpu_tdar_oracle")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results-thermogpu-methane-z"))
    parser.add_argument("--initial", type=int, default=16)
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--fill-candidates", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.budget < args.initial:
        parser.error("--budget must be >= --initial")

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    domain = ThermoGPUDomain()
    initial = farthest_point(args.initial, candidates=max(128, 8 * args.initial), seed=args.seed)
    config = TDARConfig(fill_candidates=args.fill_candidates, seed=args.seed, record_history=True)

    with ThermoGPUOracle(args.oracle, domain=domain) as oracle:
        result = sample(oracle, initial, args.budget, config)

    artifact = out / "methane-z.npz"
    save_simplicial_cpwa(result, artifact, metadata={
        "target": "methane-Z",
        "oracle": "ThermoGPU thermogpu_tdar_oracle",
        "normalization": "unit-square",
        "temperature_K": list(domain.temperature_K),
        "pressure_Pa": list(domain.pressure_Pa),
        "budget": args.budget,
        "initial_points": args.initial,
        "seed": args.seed,
    })

    with (out / "tdar-history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["iteration", "kind", "entity_index", "u", "v"])
        for step in result.history:
            for point, (kind, entity) in zip(step.proposed_points, step.selected_entities):
                writer.writerow([step.iteration, kind, entity, point[0], point[1]])

    summary = {
        "points": len(result.points),
        "simplices": len(result.simplices),
        "iterations": result.iterations,
        "temperature_K": list(domain.temperature_K),
        "pressure_Pa": list(domain.pressure_Pa),
        "artifact": artifact.name,
    }
    (out / "metadata.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
