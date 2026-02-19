"""
Reusable dataset generators for OAH experiments.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_openml


def make_noisy_xor(n: int = 400, noise: float = 0.1, random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Four Gaussian blobs arranged in an XOR pattern."""
    rng = np.random.default_rng(random_state)
    base = np.array(
        [
            [-0.5, -0.5],
            [-0.5, 0.5],
            [0.5, -0.5],
            [0.5, 0.5],
        ]
    )
    labels = np.array([0, 1, 1, 0], dtype=int)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    per_cluster = n // 4
    for center, lbl in zip(base, labels):
        pts = center + noise * rng.standard_normal(size=(per_cluster, 2))
        X_list.append(pts)
        y_list.extend([int(lbl)] * per_cluster)
    X = np.vstack(X_list)[:n]
    y = np.array(y_list, dtype=int)[:n]
    return X, y


def make_moons(n: int = 400, noise: float = 0.2, random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving moons constructed analytically (no sklearn dependency)."""
    rng = np.random.default_rng(random_state)
    n_half = n // 2
    angles = rng.uniform(0, np.pi, size=n_half)
    x1 = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    x2 = np.stack([1 - np.cos(angles), 1 - np.sin(angles)], axis=1)
    X = np.vstack([x1, x2])
    y = np.array([0] * n_half + [1] * n_half, dtype=int)
    X += noise * rng.standard_normal(size=X.shape)
    X = X[:n]
    y = y[:n]
    return X, y


def make_piecewise(n: int = 900, random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    3x3 grid with alternating labels to induce piecewise decision boundaries.
    Points sampled uniformly in [-1,1]^2.
    """
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    cell_x = ((X[:, 0] + 1.0) * 1.5).astype(int).clip(0, 2)
    cell_y = ((X[:, 1] + 1.0) * 1.5).astype(int).clip(0, 2)
    y = (cell_x + cell_y) % 2
    return X, y.astype(int)


def make_colon(random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Colon cancer microarray dataset (~62 samples, ~2000 features)."""
    rng = np.random.default_rng(random_state)
    data = fetch_openml(data_id=1432, as_frame=False)  # OpenML "colon-cancer"
    X_raw = data.data
    if hasattr(X_raw, "toarray"):
        X = X_raw.toarray().astype(float)
    else:
        X = X_raw.astype(float)
    y_raw = data.target
    classes = np.unique(y_raw)
    mapping = {cls: i for i, cls in enumerate(classes)}
    y = np.array([mapping[v] for v in y_raw], dtype=int)
    order = rng.permutation(len(X))
    return X[order], y[order]


def make_gaussians_far(
    n: int = 400,
    random_state: int = 0,
    mean_shift: float = 2.5,
    cov_scale: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated 2D Gaussians."""
    rng = np.random.default_rng(random_state)
    n_half = n // 2
    mean0 = np.array([-mean_shift, -mean_shift])
    mean1 = np.array([mean_shift, mean_shift])
    cov = cov_scale * np.eye(2)
    X0 = rng.multivariate_normal(mean0, cov, size=n_half)
    X1 = rng.multivariate_normal(mean1, cov, size=n - n_half)
    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1), dtype=int)
    order = rng.permutation(len(X))
    return X[order], y[order]


def make_gaussians_overlap(
    n: int = 400,
    random_state: int = 0,
    mean_shift: float = 0.6,
    cov_scale: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Two heavily overlapping 2D Gaussians."""
    rng = np.random.default_rng(random_state)
    n_half = n // 2
    mean0 = np.array([-mean_shift, 0.0])
    mean1 = np.array([mean_shift, 0.0])
    cov = cov_scale * np.eye(2)
    X0 = rng.multivariate_normal(mean0, cov, size=n_half)
    X1 = rng.multivariate_normal(mean1, cov, size=n - n_half)
    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1), dtype=int)
    order = rng.permutation(len(X))
    return X[order], y[order]


def make_circles(
    n: int = 400,
    noise: float = 0.08,
    factor: float = 0.5,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two concentric circles (outer radius=1, inner radius=factor) with Gaussian noise.
    """
    rng = np.random.default_rng(random_state)
    n_outer = n // 2
    n_inner = n - n_outer

    angles_outer = rng.uniform(0, 2 * np.pi, size=n_outer)
    angles_inner = rng.uniform(0, 2 * np.pi, size=n_inner)
    outer = np.stack([np.cos(angles_outer), np.sin(angles_outer)], axis=1)
    inner = factor * np.stack([np.cos(angles_inner), np.sin(angles_inner)], axis=1)

    X = np.vstack([outer, inner])
    y = np.array([0] * n_outer + [1] * n_inner, dtype=int)

    X += noise * rng.standard_normal(size=X.shape)
    order = rng.permutation(len(X))
    return X[order], y[order]


def _fetch_openml_binary(
    data_id: int, random_state: int, n_limit: int | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(random_state)
    data = fetch_openml(data_id=data_id, as_frame=False)
    X_raw = data.data
    X_arr = X_raw.toarray() if hasattr(X_raw, "toarray") else X_raw
    X = X_arr.astype(float)

    y_raw = data.target
    classes = np.unique(y_raw)
    mapping = {cls: i for i, cls in enumerate(classes)}
    y = np.array([mapping[v] for v in y_raw], dtype=int)

    order = rng.permutation(len(X))
    if n_limit is not None:
        order = order[:n_limit]
    meta = {
        "n_rows": len(X),
        "n_cols_raw": X_arr.shape[1],
        "n_cols_final": X.shape[1],
    }
    return X[order], y[order], meta


def make_spambase(random_state: int = 0, n_limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """UCI Spambase (OpenML 44): ~4601 x 57, binary."""
    return _fetch_openml_binary(44, random_state, n_limit)


def make_electricity(random_state: int = 0, n_limit: int | None = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Electricity pricing (OpenML 151), concept-drift benchmark."""
    return _fetch_openml_binary(151, random_state, n_limit)


def make_airlines(random_state: int = 0, n_limit: int | None = 50000) -> tuple[np.ndarray, np.ndarray]:
    """Airline delays (OpenML 1169), binary on-time/delay."""
    return _fetch_openml_binary(1169, random_state, n_limit)


def make_breast_cancer(random_state: int = 0, n_limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Breast Cancer Wisconsin (Diagnostic) (OpenML 1510), 569 x 30, numeric."""
    return _fetch_openml_binary(1510, random_state, n_limit)


def make_sonar(random_state: int = 0, n_limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sonar (OpenML 40), 208 x 60, numeric."""
    return _fetch_openml_binary(40, random_state, n_limit)


def make_phishing(random_state: int = 0, n_limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Phishing websites (OpenML 4534), ~11055 x 30, binary."""
    return _fetch_openml_binary(4534, random_state, n_limit)


def make_higgs_small(random_state: int = 0, n_limit: int | None = 100000) -> tuple[np.ndarray, np.ndarray]:
    """Reduced Higgs (OpenML 23512), subsample for speed."""
    return _fetch_openml_binary(23512, random_state, n_limit)


__all__ = [
    "make_noisy_xor",
    "make_moons",
    "make_piecewise",
    "make_colon",
    "make_gaussians_far",
    "make_gaussians_overlap",
    "make_circles",
    "make_spambase",
    "make_electricity",
    "make_airlines",
    "make_breast_cancer",
    "make_sonar",
    "make_phishing",
    "make_higgs_small",
]
