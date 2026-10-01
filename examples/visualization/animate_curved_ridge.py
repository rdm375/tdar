"""Generate the README curved-ridge animation from the public TDAR API."""
from pathlib import Path
import sys

import jax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from problems import curved_ridge
from tdar import TDARConfig, farthest_point, sample, jax_oracle


def main(output=None, budget=128):
    output = Path(output) if output else HERE.parents[2] / "docs" / "assets" / "curved-ridge.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    initial = farthest_point(16)
    result = sample(jax_oracle(curved_ridge), initial, budget=budget, config=TDARConfig(record_history=True))

    grid = np.linspace(0.0, 1.0, 180)
    xx, yy = np.meshgrid(grid, grid)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    zz = np.asarray(jax.vmap(curved_ridge)(xy)).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    levels = np.linspace(float(zz.min()), float(zz.max()), 22)
    ax.contourf(xx, yy, zz, levels=levels, alpha=0.72)
    ax.contour(xx, yy, zz, levels=levels[::2], linewidths=0.45, alpha=0.55)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="$x_1$", ylabel="$x_2$")
    ax.set_aspect("equal")

    mesh = LineCollection([], colors="white", linewidths=1.05, alpha=0.92)
    ax.add_collection(mesh)
    points_artist = ax.scatter([], [], s=24, c="black", edgecolors="white", linewidths=0.7, zorder=4)
    new_artist = ax.scatter([], [], s=62, marker="o", facecolors="none", edgecolors="black", linewidths=2.0, zorder=5)
    title = ax.set_title("")

    frames = list(result.history) + [None]

    def update(i):
        step = frames[i]
        if step is None:
            pts = result.points
            # final triangulation only for presentation
            from scipy.spatial import Delaunay
            simp = Delaunay(pts).simplices
            proposed = np.empty((0, 2))
            label = f"TDAR curved ridge — {len(pts)} samples"
        else:
            pts, simp, proposed = step.points, step.simplices, step.proposed_points
            label = f"TDAR curved ridge — {len(pts)} → {len(pts) + len(proposed)} samples"
        tri = pts[simp]
        segments = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
        mesh.set_segments(segments)
        points_artist.set_offsets(pts)
        new_artist.set_offsets(proposed if len(proposed) else np.empty((0, 2)))
        title.set_text(label)
        return mesh, points_artist, new_artist, title

    anim = FuncAnimation(fig, update, frames=len(frames), interval=500, blit=False, repeat_delay=1200)
    anim.save(output, writer=PillowWriter(fps=2), dpi=105)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
