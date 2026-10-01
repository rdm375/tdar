"""Generate reproducible figures and animations used by the project README."""
from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[1]))
from problems import curved_ridge
from tdar import TDARConfig, farthest_point, jax_oracle, sample

ASSETS = ROOT / "docs" / "assets"


def _save(fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)


def relu_hinge_figure():
    knots = np.array([0.18, 0.42, 0.67, 0.84])
    slopes = np.array([0.4, 2.0, -1.2, 1.4, -0.3])
    jumps = np.diff(slopes)
    x = np.linspace(0, 1, 800)
    y = 0.08 + slopes[0] * x
    for c, xi in zip(jumps, knots):
        y += c * np.maximum(x - xi, 0)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(x, y, linewidth=2.5)
    for xi in knots:
        ax.axvline(xi, linewidth=0.8, alpha=0.35)
    ax.scatter(knots, np.interp(knots, x, y), s=35, zorder=3)
    ax.set(xlabel="$x$", ylabel="$s(x)$", title="A 1-D linear interpolant is a sum of ReLU hinge functions")
    ax.grid(alpha=0.2)
    _save(fig, "relu-hinge-basis.png")


def taylor_disagreement_figure():
    f = lambda x: 0.18 + 0.35*x + 0.75*x*x + 0.18*np.sin(2*np.pi*x)
    df = lambda x: 0.35 + 1.5*x + 0.36*np.pi*np.cos(2*np.pi*x)
    x = np.linspace(0, 1, 700)
    sites = np.array([0.18, 0.52, 0.86])
    probe = 0.64
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(x, f(x), linewidth=2.5, label="$f(x)$")
    preds = []
    for xi in sites:
        tangent = f(xi) + df(xi)*(x-xi)
        ax.plot(x, tangent, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.scatter([xi], [f(xi)], s=35, zorder=4)
        preds.append(f(xi) + df(xi)*(probe-xi))
    ax.axvline(probe, linewidth=1.0, alpha=0.4)
    ax.scatter(np.full(3, probe), preds, s=48, zorder=5)
    ax.annotate("Taylor predictions\ndisagree at a common probe", xy=(probe, np.mean(preds)), xytext=(0.34, max(preds)+0.35), arrowprops={"arrowstyle":"->"})
    ax.set(xlim=(0,1), xlabel="$x$", ylabel="value", title="TDAR signal: disagreement among local first-order models")
    ax.grid(alpha=0.2)
    _save(fig, "taylor-disagreement.png")


def ridge_sampling_comparison():
    tdar = sample(jax_oracle(curved_ridge), farthest_point(16), budget=128)
    rng = np.random.default_rng(20261001)
    uniform = np.vstack([np.array([[0,0],[1,0],[0,1],[1,1]], float), rng.random((124,2))])
    grid = np.linspace(0,1,220); xx, yy = np.meshgrid(grid, grid)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    zz = np.asarray(jax.vmap(curved_ridge)(jnp.asarray(xy))).reshape(xx.shape)
    fig, axes = plt.subplots(1,2,figsize=(10.4,4.8), sharex=True, sharey=True)
    levels=np.linspace(float(zz.min()),float(zz.max()),24)
    for ax, pts, title in zip(axes,[uniform,tdar.points],["Nonadaptive coverage (128 locations)","TDAR (128 locations)"]):
        ax.contourf(xx,yy,zz,levels=levels,alpha=0.65)
        tri=Delaunay(pts)
        ax.triplot(pts[:,0],pts[:,1],tri.simplices,linewidth=0.45,alpha=0.8)
        ax.scatter(pts[:,0],pts[:,1],s=7)
        ax.set(title=title,xlabel="$x_1$",aspect="equal")
    axes[0].set_ylabel("$x_2$")
    _save(fig,"ridge-sampling-comparison.png")


def piecewise_affine_surface():
    result=sample(jax_oracle(curved_ridge),farthest_point(16),budget=96)
    tri=Delaunay(result.points)
    interp=LinearNDInterpolator(tri,result.values)
    grid=np.linspace(0,1,100); xx,yy=np.meshgrid(grid,grid); xy=np.column_stack([xx.ravel(),yy.ravel()]); zz=interp(xy).reshape(xx.shape)
    fig=plt.figure(figsize=(8.2,5.8)); ax=fig.add_subplot(111,projection="3d")
    ax.plot_surface(xx,yy,zz,linewidth=0,antialiased=True,alpha=0.82)
    ax.plot_trisurf(result.points[:,0],result.points[:,1],result.values,triangles=tri.simplices,alpha=0.18,edgecolor="k",linewidth=0.25)
    ax.set(xlabel="$x_1$",ylabel="$x_2$",zlabel="$I_h f$",title="The sampled values define a continuous piecewise-affine interpolant")
    ax.view_init(elev=30,azim=-55)
    _save(fig,"piecewise-affine-surface.png")


def refinement_step_animation():
    result=sample(jax_oracle(curved_ridge),farthest_point(16),budget=80,config=TDARConfig(batch_size=1,record_history=True))
    # Pick a visually informative later step.
    indices=np.linspace(0,len(result.history)-1,8,dtype=int)
    fig,ax=plt.subplots(figsize=(6.2,5.6)); ax.set(xlim=(0,1),ylim=(0,1),xlabel="$x_1$",ylabel="$x_2$",aspect="equal")
    mesh=LineCollection([],linewidths=0.8,alpha=0.8); ax.add_collection(mesh)
    pts_artist=ax.scatter([],[],s=16,zorder=3); new_artist=ax.scatter([],[],s=85,facecolors="none",edgecolors="black",linewidths=2,zorder=4)
    title=ax.set_title("")
    def update(k):
        step=result.history[indices[k]]; pts=step.points; simp=step.simplices; tri=pts[simp]
        seg=np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]],axis=0); mesh.set_segments(seg); pts_artist.set_offsets(pts); new_artist.set_offsets(step.proposed_points)
        kind=step.selected_entities[0][0] if step.selected_entities else "fill"
        title.set_text(f"One TDAR decision: {len(pts)} samples — next point from {kind}")
        return mesh,pts_artist,new_artist,title
    anim=FuncAnimation(fig,update,frames=len(indices),interval=850,repeat_delay=1300)
    path=ASSETS/"refinement-decisions.gif"; anim.save(path,writer=PillowWriter(fps=1.3),dpi=105); plt.close(fig); print(path)


def construction_pipeline_animation():
    result=sample(jax_oracle(curved_ridge),farthest_point(16),budget=96)
    tri=Delaunay(result.points); interp=LinearNDInterpolator(tri,result.values)
    grid=np.linspace(0,1,180); xx,yy=np.meshgrid(grid,grid); xy=np.column_stack([xx.ravel(),yy.ravel()])
    true=np.asarray(jax.vmap(curved_ridge)(jnp.asarray(xy))).reshape(xx.shape); approx=interp(xy).reshape(xx.shape)
    fig,ax=plt.subplots(figsize=(7.3,5.7)); levels=np.linspace(float(true.min()),float(true.max()),24)
    stages=["target","samples","mesh","interpolant","dense"]
    def update(i):
        ax.clear(); stage=stages[i]; ax.set(xlim=(0,1),ylim=(0,1),xlabel="$x_1$",ylabel="$x_2$",aspect="equal")
        if stage=="target":
            ax.contourf(xx,yy,true,levels=levels); ax.set_title("1. Expensive differentiable target")
        elif stage=="samples":
            ax.contourf(xx,yy,true,levels=levels,alpha=.45); ax.scatter(result.points[:,0],result.points[:,1],s=14); ax.set_title("2. TDAR concentrates first-order samples")
        elif stage=="mesh":
            ax.triplot(result.points[:,0],result.points[:,1],tri.simplices,linewidth=.7); ax.scatter(result.points[:,0],result.points[:,1],s=10); ax.set_title("3. Samples induce a piecewise-affine mesh")
        elif stage=="interpolant":
            ax.contourf(xx,yy,approx,levels=levels); ax.triplot(result.points[:,0],result.points[:,1],tri.simplices,linewidth=.35,alpha=.5); ax.set_title("4. Continuous piecewise-affine interpolant")
        else:
            ax.imshow(approx,origin="lower",extent=(0,1,0,1),aspect="equal"); ax.set_title("5. ReLU form: regular dense accelerator evaluation")
        return []
    anim=FuncAnimation(fig,update,frames=len(stages),interval=1200,repeat_delay=1800)
    path=ASSETS/"construction-pipeline.gif"; anim.save(path,writer=PillowWriter(fps=.8),dpi=105); plt.close(fig); print(path)


def main():
    relu_hinge_figure(); taylor_disagreement_figure(); ridge_sampling_comparison(); piecewise_affine_surface(); refinement_step_animation(); construction_pipeline_animation()

if __name__=="__main__": main()
