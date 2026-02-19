from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path


def _to_pm1(y: int) -> int:
    return 1 if y == 1 else -1


class KernelPerceptron:
    """Simple kernel perceptron with RBF kernel."""

    def __init__(self, gamma: float = 2.0) -> None:
        self.gamma = gamma
        self.support_X: list[np.ndarray] = []
        self.support_y: list[int] = []
        self.alpha: list[float] = []

    def _k(self, x: np.ndarray, z: np.ndarray) -> float:
        diff = x - z
        return float(np.exp(-self.gamma * np.dot(diff, diff)))

    def predict(self, x: np.ndarray) -> int:
        if not self.support_X:
            return 0
        s = 0.0
        for a, y_i, x_i in zip(self.alpha, self.support_y, self.support_X):
            s += a * y_i * self._k(x, x_i)
        return 1 if s >= 0 else 0

    def partial_fit(self, x: np.ndarray, y: np.ndarray, classes: np.ndarray | None = None) -> None:
        y_pm1 = _to_pm1(int(y[0]))
        margin = 0.0
        for a, y_i, x_i in zip(self.alpha, self.support_y, self.support_X):
            margin += a * y_i * self._k(x[0], x_i)
        if y_pm1 * margin <= 0.0:
            self.support_X.append(x[0].copy())
            self.support_y.append(y_pm1)
            self.alpha.append(1.0)


class KernelPA:
    """Kernel passive-aggressive with RBF kernel."""

    def __init__(self, gamma: float = 2.0) -> None:
        self.gamma = gamma
        self.support_X: list[np.ndarray] = []
        self.support_y: list[int] = []
        self.alpha: list[float] = []

    def _k(self, x: np.ndarray, z: np.ndarray) -> float:
        diff = x - z
        return float(np.exp(-self.gamma * np.dot(diff, diff)))

    def predict(self, x: np.ndarray) -> int:
        if not self.support_X:
            return 0
        s = 0.0
        for a, y_i, x_i in zip(self.alpha, self.support_y, self.support_X):
            s += a * y_i * self._k(x, x_i)
        return 1 if s >= 0 else 0

    def partial_fit(self, x: np.ndarray, y: np.ndarray, classes: np.ndarray | None = None) -> None:
        y_pm1 = _to_pm1(int(y[0]))
        margin = 0.0
        for a, y_i, x_i in zip(self.alpha, self.support_y, self.support_X):
            margin += a * y_i * self._k(x[0], x_i)
        loss = max(0.0, 1.0 - y_pm1 * margin)
        if loss > 0.0:
            tau = loss
            self.support_X.append(x[0].copy())
            self.support_y.append(y_pm1)
            self.alpha.append(tau)

def plot_histories(
    xs: np.ndarray,
    history_pre: dict[str, list[float]],
    history_rem: dict[str, list[float]],
    dataset: str,
    plot_path: str,
    max_rem_steps: int | None = None,
) -> None:
    """Plot prequential and remaining-set accuracy histories."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    label_map = {"oah": "OAH (ours)", "v1": "OAH (ours)", "OAH": "OAH (ours)"}
    for name, series in history_pre.items():
        axes[0].plot(xs, series, label=label_map.get(name, name))
    axes[0].set_xlabel("Samples seen")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"Prequential — {dataset}")

    if max_rem_steps is None:
        xs_rem = xs
    else:
        limit = min(max_rem_steps, len(xs))
        xs_rem = xs[:limit]
    for name, series in history_rem.items():
        axes[1].plot(xs_rem, series[: len(xs_rem)], label=label_map.get(name, name))
    axes[1].set_xlabel("Samples seen")
    axes[1].set_title(f"Remaining set — {dataset}")
    if len(xs_rem) > 0:
        axes[1].set_xlim(xs_rem[0], xs_rem[-1])
    axes[1].legend()
    fig.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


def print_table(datasets: list[str], results: dict[str, dict[str, float]]) -> None:
    """Pretty-print prequential accuracy table."""
    methods = sorted(next(iter(results.values())).keys())
    # normalize naming
    method_labels = {"oah": "OAH (ours)", "v1": "OAH (ours)", "OAH": "OAH (ours)"}
    header = ["method"] + datasets
    max_method_len = max(len(m) for m in methods + ["method"])
    col_widths = [max(max_method_len + 3, len("method") + 3)]
    col_widths += [max(len(h), 10) for h in datasets]

    def fmt_row(values: list[str]) -> str:
        return " | ".join(v.ljust(w) for v, w in zip(values, col_widths))

    print("\nPrequential accuracy table:")
    print(fmt_row(header))
    print("-+-".join("-" * w for w in col_widths))
    for m in methods:
        label = method_labels.get(m, m)
        row = [label] + [f"{results[d][m]:.3f}" for d in datasets]
        print(fmt_row(row))


def aggregate_results(results_list: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    """Compute mean/std for a list of metric dicts keyed by method."""
    if not results_list:
        return {}, {}
    methods = results_list[0].keys()
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for m in methods:
        vals = np.array([r[m] for r in results_list], dtype=float)
        mean[m] = float(np.mean(vals))
        std[m] = float(np.std(vals))
    return mean, std


def print_table_stats(
    title: str,
    datasets: list[str],
    mean_results: dict[str, dict[str, float]],
    std_results: dict[str, dict[str, float]],
    value_label: str,
) -> None:
    """Pretty-print mean±std table across datasets."""
    methods = sorted(next(iter(mean_results.values())).keys())
    method_labels = {"oah": "OAH (ours)", "v1": "OAH (ours)", "OAH": "OAH (ours)"}
    header = ["method"] + datasets
    max_method_len = max(len(m) for m in methods + ["method"])
    col_widths = [max(max_method_len + 3, len("method") + 3)]
    col_widths += [max(len(h), len(value_label), 10) for h in datasets]

    def fmt_row(values: list[str]) -> str:
        return " | ".join(v.ljust(w) for v, w in zip(values, col_widths))

    print(f"\n{title}:")
    print(fmt_row(header))
    print("-+-".join("-" * w for w in col_widths))
    for m in methods:
        label = method_labels.get(m, m)
        row = [label]
        for d in datasets:
            mean = mean_results[d][m]
            std = std_results[d][m]
            val = f"{mean*100:.1f}±{std*100:.1f}" if np.isfinite(mean) and np.isfinite(std) else "nan"
            row.append(val)
        print(fmt_row(row))

def compute_aulc(history_rem: dict[str, list[float]], max_steps: int = 20) -> dict[str, float]:
    """Compute remaining-set accuracy after the first N observations (ISA@N)."""
    scores: dict[str, float] = {}
    for name, series in history_rem.items():
        if not series:
            scores[name] = float("nan")
            continue
        if len(series) >= max_steps:
            scores[name] = float(series[max_steps - 1])
        else:
            scores[name] = float(series[-1])
    return scores


def print_aulc_table(dataset: str, aulc: dict[str, float]) -> None:
    """Pretty-print ISA@N table for a single dataset."""
    method_labels = {"oah": "OAH (ours)", "v1": "OAH (ours)", "OAH": "OAH (ours)"}
    methods = sorted(aulc.keys())
    max_method_len = max(len(m) for m in methods + ["method"])
    col_widths = [max(max_method_len + 3, len("method") + 3), max(len("ISA@20"), 10)]

    def fmt_row(values: list[str]) -> str:
        return " | ".join(v.ljust(w) for v, w in zip(values, col_widths))

    print(f"\nISA@20 (remaining set) — {dataset}:")
    print(fmt_row(["method", "ISA@20"]))
    print("-+-".join("-" * w for w in col_widths))
    for m in methods:
        label = method_labels.get(m, m)
        val = aulc[m]
        row = [label, f"{val*100:.1f}" if np.isfinite(val) else "nan"]
        print(fmt_row(row))


def print_aulc_table_multi(datasets: list[str], results: dict[str, dict[str, float]], max_steps: int = 20) -> None:
    """Pretty-print ISA@N table across datasets."""
    methods = sorted(next(iter(results.values())).keys())
    method_labels = {"oah": "OAH (ours)", "v1": "OAH (ours)", "OAH": "OAH (ours)"}
    header = ["method"] + datasets
    max_method_len = max(len(m) for m in methods + ["method"])
    col_widths = [max(max_method_len + 3, len("method") + 3)]
    col_widths += [max(len(h), 10) for h in datasets]

    def fmt_row(values: list[str]) -> str:
        return " | ".join(v.ljust(w) for v, w in zip(values, col_widths))


def save_results_csv(
    output_path: Path,
    datasets: list[str],
    metrics: list[tuple[str, dict[str, dict[str, float]], dict[str, dict[str, float]]]],
    method_labels: dict[str, str],
) -> None:
    """Save mean/std results to CSV for multiple metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "metric", "method", "mean", "std"])
        for metric_name, mean_results, std_results in metrics:
            methods = sorted(next(iter(mean_results.values())).keys())
            for d in datasets:
                for m in methods:
                    label = method_labels.get(m, m)
                    mean = mean_results[d][m]
                    std = std_results[d][m]
                    if metric_name == "K":
                        writer.writerow([d, metric_name, label, mean, std])
                    else:
                        writer.writerow([d, metric_name, label, mean * 100.0, std * 100.0])

    # No printing here; this helper only saves results to CSV.


def print_oah_summary(datasets: list[str], Ks: dict[str, str], metas: dict[str, dict[str, str | int]]) -> None:
    """Print summary of OAH model size and feature counts."""
    header = ["dataset", "K (mean±std)", "cols", "rows total", "rows used", "majority% used"]
    max_dataset_len = max(len(d) for d in datasets + ["dataset"])
    col_widths = [
        max_dataset_len + 3,
        max(len("K (mean±std)"), 12),
        max(len("cols"), 4),
        max(len("rows total"), 10),
        max(len("rows used"), 9),
        max(len("majority% used"), 14),
    ]

    def fmt_row(values: list[str]) -> str:
        return " | ".join(v.ljust(w) for v, w in zip(values, col_widths))

    print("\nOAH summary:")
    print(fmt_row(header))
    print("-+-".join("-" * w for w in col_widths))
    for d in datasets:
        meta = metas.get(d, {})
        row = [
            d,
            str(Ks.get(d, "")),
            str(meta.get("n_cols_final", "")),
            str(meta.get("n_rows_total", "")),
            str(meta.get("n_rows_used", "")),
            str(meta.get("majority_pct_used", "")),
        ]
        print(fmt_row(row))


# Backward-compatible alias after v1 -> oah rename.
print_v1_summary = print_oah_summary
