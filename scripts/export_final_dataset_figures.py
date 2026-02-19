"""
Export final OAH partition plots for 2D datasets used in the paper.

This script reproduces the "Go to End" state from GUI-like settings and saves
figures directly to article/llncs/figures/datasets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from oah import OnlineAdditiveHyperplanes
from oah.datasets import (
    make_circles,
    make_gaussians_far,
    make_gaussians_overlap,
    make_moons,
    make_noisy_xor,
    make_piecewise,
)


C_CLASS0 = "#0072B2"
C_CLASS1 = "#D55E00"
C_HPLANE = "#4D4D4D"
C_GRID = "#BDBDBD"
CMAP_BG = ListedColormap(["#BDD7EE", "#F4C7B5"])


def _scale_to_minus_one_one(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    span = np.where((maxs - mins) > 1e-12, (maxs - mins), 1.0)
    return 2.0 * (X - mins) / span - 1.0


def _dataset_spec(name: str):
    if name == "xor":
        seed = 123
        X, y = make_noisy_xor(n=400, noise=0.10, random_state=seed)
    elif name == "moons":
        seed = 456
        X, y = make_moons(n=400, noise=0.20, random_state=seed)
    elif name == "circles":
        seed = 654
        X, y = make_circles(n=400, noise=0.08, random_state=seed)
    elif name == "piecewise":
        seed = 789
        X, y = make_piecewise(n=900, random_state=seed)
    elif name == "gauss_far":
        seed = 111
        X, y = make_gaussians_far(n=400, random_state=seed)
    elif name == "gauss_overlap":
        seed = 222
        X, y = make_gaussians_overlap(n=400, random_state=seed)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return seed, _scale_to_minus_one_one(np.asarray(X, dtype=float)), y.astype(int)


def _square_bounds(X: np.ndarray) -> tuple[float, float, float, float]:
    x_min, y_min = np.min(X, axis=0)
    x_max, y_max = np.max(X, axis=0)
    dx = max(1e-9, x_max - x_min)
    dy = max(1e-9, y_max - y_min)
    d = max(dx, dy)
    margin = 0.08 * d
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * d + margin
    return cx - half, cx + half, cy - half, cy + half


def _train_full(X: np.ndarray, y: np.ndarray, seed: int) -> OnlineAdditiveHyperplanes:
    b = OnlineAdditiveHyperplanes(random_state=seed)
    order = np.random.default_rng(seed).permutation(len(X))
    for idx in order:
        b.add_point(X[int(idx)], int(y[int(idx)]))
    return b


def _draw_background(ax, b: OnlineAdditiveHyperplanes, bounds: tuple[float, float, float, float]) -> None:
    if not b.cells:
        return
    xmin, xmax, ymin, ymax = bounds
    grid_res = 300
    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    coords = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    preds = np.array([b.predict(pt) for pt in coords], dtype=float).reshape(grid_res, grid_res)
    im = ax.imshow(
        preds,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap=CMAP_BG,
        vmin=0,
        vmax=1,
        alpha=0.20,
        interpolation="nearest",
    )
    im.cmap.set_bad(alpha=0.0)


def _plot_one(name: str, out_path: Path) -> None:
    seed, X, y = _dataset_spec(name)
    b = _train_full(X, y, seed)
    bounds = _square_bounds(X)
    xmin, xmax, ymin, ymax = bounds

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=300)
    _draw_background(ax, b, bounds)

    mask0 = y == 0
    mask1 = y == 1
    if np.any(mask0):
        ax.scatter(X[mask0, 0], X[mask0, 1], c=C_CLASS0, s=24, edgecolors="k", linewidths=0.3, label="class 0")
    if np.any(mask1):
        ax.scatter(X[mask1, 0], X[mask1, 1], c=C_CLASS1, s=24, edgecolors="k", linewidths=0.3, label="class 1")

    for hp in b.hyperplanes:
        n = hp.normal
        if len(n) != 2:
            continue
        if abs(n[1]) < 1e-8:
            x_line = hp.bias / (n[0] + 1e-12)
            ax.plot([x_line, x_line], [ymin, ymax], linestyle="--", color=C_HPLANE, alpha=0.9, linewidth=1.2)
        else:
            xs = np.linspace(xmin, xmax, 200)
            ys = (hp.bias - n[0] * xs) / (n[1] + 1e-12)
            ax.plot(xs, ys, linestyle="--", color=C_HPLANE, alpha=0.9, linewidth=1.2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(name.replace("_", " ").title())
    ax.grid(True, color=C_GRID, alpha=0.35, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path("article/llncs/figures/datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["xor", "moons", "circles", "gauss_far", "gauss_overlap", "piecewise"]
    for name in names:
        out_path = out_dir / f"oah_plot_{name}.png"
        _plot_one(name, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

