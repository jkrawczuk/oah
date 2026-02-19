"""
Evaluate OnlineAdditiveHyperplanes (OAH) with different parameter sets on multiple datasets.

Datasets:
- xor_noise0.10
- moons_noise0.20
- circles_noise0.08
- gaussians_far
- gaussians_overlap
- piecewise_3x3
- colon (OpenML 1432, ~62x2000)

Metrics:
- prequential accuracy (predict-then-update)
- final number of hyperplanes K

Results are printed as a table and saved to CSV.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from tqdm import tqdm

from oah import OnlineAdditiveHyperplanes
from oah.datasets import (
    make_colon,
    make_spambase,
    make_electricity,
    make_phishing,
    make_breast_cancer,
    make_sonar,
    make_circles,
    make_gaussians_far,
    make_gaussians_overlap,
    make_moons,
    make_noisy_xor,
    make_piecewise,
)


DatasetGen = Callable[[int], Tuple[np.ndarray, np.ndarray]]


def prequential_accuracy(
    builder: OnlineAdditiveHyperplanes, X: np.ndarray, y: np.ndarray, random_state: int, burn_in_prequential: int = 1
) -> Tuple[float, int, int, int, int, int, int, int]:
    correct = 0
    eval_total = 0
    block_patience = 0
    block_maxk = 0
    block_stable = 0
    block_other = 0
    block_single_class = 0
    block_low_gain = 0
    block_redundant = 0
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(X))
    for step, idx in enumerate(order):
        eval_active = step >= burn_in_prequential
        if eval_active:
            eval_total += 1
        pred = builder.predict(X[idx])
        if eval_active and pred == int(y[idx]):
            correct += 1
        info = builder.add_point(X[idx], int(y[idx]))
        if not info.get("grew"):
            if info.get("reason") == "patience":
                block_patience += 1
            elif info.get("reason") == "max_K_reached":
                block_maxk += 1
            elif info.get("reason") == "stable_cell":
                block_stable += 1
            elif info.get("reason") == "single_class_cell":
                block_single_class += 1
            elif info.get("reason") in {"degenerate_pair", "low_impurity_gain"}:
                block_low_gain += 1
            elif info.get("reason") == "redundant":
                block_redundant += 1
            else:
                block_other += 1
    acc = correct / eval_total if eval_total > 0 else 0.0
    return (
        acc,
        len(builder.hyperplanes),
        block_patience,
        block_maxk,
        block_stable,
        block_single_class,
        block_low_gain,
        block_redundant,
        block_other,
    )


def main() -> None:
    datasets: Dict[str, DatasetGen] = {
        "xor_noise0.10": lambda seed: make_noisy_xor(n=400, noise=0.10, random_state=seed),
        "moons_noise0.20": lambda seed: make_moons(n=400, noise=0.20, random_state=seed),
        "circles_noise0.08": lambda seed: make_circles(n=400, noise=0.08, random_state=seed),
        "gaussians_far": lambda seed: make_gaussians_far(n=400, random_state=seed),
        "gaussians_overlap": lambda seed: make_gaussians_overlap(n=400, random_state=seed),
        "piecewise_3x3": lambda seed: make_piecewise(n=900, random_state=seed),
        "colon": lambda seed: make_colon(random_state=seed),
        "spambase": lambda seed: make_spambase(random_state=seed, n_limit=1000)[:2],
        "electricity": lambda seed: make_electricity(random_state=seed, n_limit=1000)[:2],
        "phishing": lambda seed: make_phishing(random_state=seed, n_limit=1000)[:2],
        "breast_cancer": lambda seed: make_breast_cancer(random_state=seed, n_limit=1000)[:2],
        "sonar": lambda seed: make_sonar(random_state=seed, n_limit=1000)[:2],
    }

    @dataclass
    class Variant:
        name: str
        params: Dict[str, object]

    variants = [
        Variant("patience_auto", {}),
        Variant("patience_1", {"grow_patience": 1}),
        Variant("patience_3", {"grow_patience": 3}),
        Variant(
            "simple_min",
            {
                "grow_patience": 1,
                "split_criterion": "midpoint",
                "min_impurity_drop": 0.0,
                "redundant_cos_thresh": 1.0,
                "redundant_bias_frac": 0.0,
            },
        ),
        Variant("anchor_centroid", {"anchor_strategy": "centroid"}),
        Variant("anchor_farthest", {"anchor_strategy": "farthest"}),
        Variant("drop_0.00", {"min_impurity_drop": 0.00}),
        Variant("drop_0.02", {"min_impurity_drop": 0.02}),
        Variant("drop_0.10", {"min_impurity_drop": 0.10}),
        Variant("redcos_0.95", {"redundant_cos_thresh": 0.95}),
        Variant("redbias_0.01", {"redundant_bias_frac": 0.01}),
        Variant("redbias_0.05", {"redundant_bias_frac": 0.05}),
    ]

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    n_repeats = 10
    base_seed = 1000

    total_runs = len(datasets) * len(variants) * n_repeats
    with tqdm(total=total_runs, desc="ablation runs") as pbar:
        for dname, gen in datasets.items():
            res_variant: Dict[str, Dict[str, float]] = {}
            for variant in variants:
                acc_runs = []
                k_runs = []
                block_patience_runs = []
                block_maxk_runs = []
                block_stable_runs = []
                block_single_runs = []
                block_low_gain_runs = []
                block_redundant_runs = []
                block_other_runs = []
                for r in range(n_repeats):
                    seed = base_seed + r * 17
                    X, y = gen(seed)
                    b = OnlineAdditiveHyperplanes(random_state=seed, **variant.params)
                    (
                        acc,
                        K,
                        block_patience,
                        block_maxk,
                        block_stable,
                        block_single_class,
                        block_low_gain,
                        block_redundant,
                        block_other,
                    ) = prequential_accuracy(b, X, y, random_state=seed)
                    acc_runs.append(acc)
                    k_runs.append(K)
                    block_patience_runs.append(block_patience)
                    block_maxk_runs.append(block_maxk)
                    block_stable_runs.append(block_stable)
                    block_single_runs.append(block_single_class)
                    block_low_gain_runs.append(block_low_gain)
                    block_redundant_runs.append(block_redundant)
                    block_other_runs.append(block_other)
                    pbar.update(1)
                res_variant[variant.name] = {
                    "acc_mean": float(np.mean(acc_runs)),
                    "acc_std": float(np.std(acc_runs)),
                    "K_mean": float(np.mean(k_runs)),
                    "K_std": float(np.std(k_runs)),
                    "block_patience_mean": float(np.mean(block_patience_runs)),
                    "block_patience_std": float(np.std(block_patience_runs)),
                    "block_maxk_mean": float(np.mean(block_maxk_runs)),
                    "block_maxk_std": float(np.std(block_maxk_runs)),
                    "block_stable_mean": float(np.mean(block_stable_runs)),
                    "block_stable_std": float(np.std(block_stable_runs)),
                    "block_single_class_mean": float(np.mean(block_single_runs)),
                    "block_single_class_std": float(np.std(block_single_runs)),
                    "block_low_gain_mean": float(np.mean(block_low_gain_runs)),
                    "block_low_gain_std": float(np.std(block_low_gain_runs)),
                    "block_redundant_mean": float(np.mean(block_redundant_runs)),
                    "block_redundant_std": float(np.std(block_redundant_runs)),
                    "block_other_mean": float(np.mean(block_other_runs)),
                    "block_other_std": float(np.std(block_other_runs)),
                }
            results[dname] = res_variant

    def print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
        widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(header)]
        print(f"\n{title}")
        header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(header))
        print(header_line)
        print("-+-".join("-" * w for w in widths))
        for row in rows:
            print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))

    # Prequential accuracy table
    header = ["dataset"] + [v.name for v in variants]
    rows = []
    for dname in datasets:
        row = [dname] + [
            f"{results[dname][v.name]['acc_mean']*100:.1f}±{results[dname][v.name]['acc_std']*100:.1f}"
            for v in variants
        ]
        rows.append(row)
    print_table("Prequential accuracy:", header, rows)

    # Final K table
    k_header = ["dataset"] + [v.name for v in variants]
    k_rows = []
    for dname in datasets:
        row = [
            dname
        ] + [
            f"{results[dname][v.name]['K_mean']:.1f}±{results[dname][v.name]['K_std']:.1f}"
            for v in variants
        ]
        k_rows.append(row)
    print_table("Final K (number of hyperplanes):", k_header, k_rows)

    # Block reasons table (per dataset/variant)
    block_header = [
        "dataset",
        "variant",
        "block_stable",
        "block_patience",
        "block_maxK",
        "block_single",
        "block_low_gain",
        "block_redundant",
        "block_other",
    ]
    block_rows = []
    for dname in datasets:
        for v in variants:
            r = results[dname][v.name]
            block_rows.append(
                [
                    dname,
                    v.name,
                    f"{r['block_stable_mean']:.1f}±{r['block_stable_std']:.1f}",
                    f"{r['block_patience_mean']:.1f}±{r['block_patience_std']:.1f}",
                    f"{r['block_maxk_mean']:.1f}±{r['block_maxk_std']:.1f}",
                    f"{r['block_single_class_mean']:.1f}±{r['block_single_class_std']:.1f}",
                    f"{r['block_low_gain_mean']:.1f}±{r['block_low_gain_std']:.1f}",
                    f"{r['block_redundant_mean']:.1f}±{r['block_redundant_std']:.1f}",
                    f"{r['block_other_mean']:.1f}±{r['block_other_std']:.1f}",
                ]
            )
    print_table("Block reasons (counts):", block_header, block_rows)

    # Save CSV
    out_path = Path("results/oah_ablation.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "variant",
                "acc_mean",
                "acc_std",
                "K_mean",
                "K_std",
                "block_stable_mean",       # stable_cell check happens before conflicts accumulate
                "block_stable_std",
                "block_patience_mean",     # conflict counter below local interval
                "block_patience_std",
                "block_maxK_mean",         # limit on hyperplanes
                "block_maxK_std",
                "block_single_class_mean", # cell has only one class
                "block_single_class_std",
                "block_low_gain_mean",     # degenerate or too low impurity gain
                "block_low_gain_std",
                "block_redundant_mean",    # candidate too similar to existing
                "block_redundant_std",
                "block_other_mean",
                "block_other_std",
            ]
        )
        for dname in datasets:
            for v in variants:
                r = results[dname][v.name]
                writer.writerow(
                    [
                        dname,
                        v.name,
                        f"{r['acc_mean']:.6f}",
                        f"{r['acc_std']:.6f}",
                        f"{r['K_mean']:.6f}",
                        f"{r['K_std']:.6f}",
                        f"{r['block_stable_mean']:.6f}",
                        f"{r['block_stable_std']:.6f}",
                        f"{r['block_patience_mean']:.6f}",
                        f"{r['block_patience_std']:.6f}",
                        f"{r['block_maxk_mean']:.6f}",
                        f"{r['block_maxk_std']:.6f}",
                        f"{r['block_single_class_mean']:.6f}",
                        f"{r['block_single_class_std']:.6f}",
                        f"{r['block_low_gain_mean']:.6f}",
                        f"{r['block_low_gain_std']:.6f}",
                        f"{r['block_redundant_mean']:.6f}",
                        f"{r['block_redundant_std']:.6f}",
                        f"{r['block_other_mean']:.6f}",
                        f"{r['block_other_std']:.6f}",
                    ]
                )
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
