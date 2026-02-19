"""
Compare the OAH OnlineAdditiveHyperplanes against simple online baselines on several datasets.

Models:
- OnlineAdditiveHyperplanes (oah)
- sklearn Perceptron
- sklearn SGDClassifier (hinge)
- sklearn SGDClassifier (hinge, PA-style)
- river LogisticRegression
- river HoeffdingTreeClassifier

For each method we track online accuracy and print a short summary.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import argparse
import concurrent.futures as cf
import multiprocessing as mp
import time
import sys
from sklearn.linear_model import Perceptron, SGDClassifier
from tqdm import tqdm
from river import linear_model as rv_linear
from river import ensemble as rv_ens
from river import tree as rv_tree

from oah import OnlineAdditiveHyperplanes
from oah.datasets import (
    make_circles,
    make_gaussians_far,
    make_gaussians_overlap,
    make_moons,
    make_noisy_xor,
    make_piecewise,
)
# allow running as a script (add repo root to sys.path)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from .compare_utils import (
        KernelPA,
        KernelPerceptron,
        aggregate_results,
        compute_aulc,
        plot_histories,
        print_table_stats,
        print_oah_summary,
        save_results_csv,
    )
except Exception:
    from scripts.compare_utils import (  # type: ignore
        KernelPA,
        KernelPerceptron,
        aggregate_results,
        compute_aulc,
        plot_histories,
        print_table_stats,
        print_oah_summary,
        save_results_csv,
    )

def run_online_eval(
    n: int = 400,
    noise: float = 0.12,
    random_state: int = 123,
    dataset: str = "xor",
    make_plot: bool = False,
    plot_path: str = "scripts/img/compare_baselines_accuracy.png",
    forward_steps: int = 20,
    burn_in_prequential: int = 1,
    progress_bar: tqdm | None = None,
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, list[float]], int]:
    if dataset == "xor":
        X, y = make_noisy_xor(n=n, noise=noise, random_state=random_state)
    elif dataset == "moons":
        X, y = make_moons(n=n, noise=noise, random_state=random_state)
    elif dataset == "piecewise":
        X, y = make_piecewise(n=n, random_state=random_state)
    elif dataset == "gauss_far":
        X, y = make_gaussians_far(n=n, random_state=random_state)
    elif dataset == "gauss_overlap":
        X, y = make_gaussians_overlap(n=n, random_state=random_state)
    elif dataset == "circles":
        X, y = make_circles(n=n, noise=noise, random_state=random_state)
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(X))
    total_samples = len(order)

    builder = OnlineAdditiveHyperplanes(random_state=random_state)

    baselines = {
        "Perceptron": Perceptron(random_state=random_state, warm_start=True, max_iter=1, tol=None),
        "SGD-Hinge": SGDClassifier(random_state=random_state, warm_start=True, max_iter=1, tol=None, loss="hinge"),
        "SGD-PA": SGDClassifier(
            random_state=random_state,
            warm_start=True,
            max_iter=1,
            tol=None,
            loss="hinge",
            learning_rate="pa1",
            eta0=1.0,
            penalty=None,
        ),
    }
    fitted = {name: False for name in baselines}
    kernel_models = {
        "K-Perceptron": KernelPerceptron(gamma=3.0),
        "K-PA": KernelPA(gamma=3.0),
    }
    river_models = {
        "LogReg": rv_linear.LogisticRegression(),
        "HoeffdingTree": rv_tree.HoeffdingTreeClassifier(
            grace_period=2,
            max_depth=25,
            delta=1e-7,
            tau=0.0,
            split_criterion="gini",
            leaf_prediction="nb",
        ),
        "OzaBagging": rv_ens.BaggingClassifier(
            model=rv_tree.HoeffdingTreeClassifier(
                grace_period=5,
                max_depth=25,
                delta=1e-7,
                tau=0.0,
                split_criterion="gini",
                leaf_prediction="nb",
            ),
            n_models=10,
            seed=random_state,
        ),
        "LeveragingBag": rv_ens.LeveragingBaggingClassifier(
            model=rv_tree.HoeffdingTreeClassifier(
                grace_period=5,
                max_depth=20,
                delta=1e-7,
                tau=0.0,
                split_criterion="gini",
                leaf_prediction="nb",
            ),
            n_models=10,
            w=6.0,
            seed=random_state,
        ),
    }

    correct = {
        "oah": 0,
        "Majority": 0,
        "1-NN": 0,
        **{name: 0 for name in baselines},
        **{name: 0 for name in kernel_models},
        **{name: 0 for name in river_models},
    }
    history_pre = {name: [] for name in correct}
    history_rem = {name: [] for name in correct}
    total = 0
    eval_total = 0
    seen_0 = 0
    seen_1 = 0

    if dataset in {"xor", "moons", "piecewise", "gauss_far", "gauss_overlap", "circles"}:
        eval_points = set(range(min(40, total_samples)))  # dense at start
    else:
        eval_points = set(np.unique(np.linspace(0, total_samples - 1, num=min(10, total_samples), dtype=int)))
    eval_points.update(range(min(forward_steps, total_samples)))

    seen_X: list[np.ndarray] = []
    seen_y: list[int] = []
    for step, idx in enumerate(order):
        x_i = X[idx].reshape(1, -1)
        y_i = np.array([y[idx]])
        eval_active = step >= burn_in_prequential
        if eval_active:
            eval_total += 1

        # Predict before updating (online evaluation).
        if seen_1 > seen_0:
            majority_pred = 1
        else:
            majority_pred = 0
        if eval_active and majority_pred == y_i[0]:
            correct["Majority"] += 1

        if seen_X:
            dists = np.linalg.norm(np.vstack(seen_X) - x_i[0], axis=1)
            nn_idx = int(np.argmin(dists))
            nn_pred = int(seen_y[nn_idx])
        else:
            nn_pred = 0
        if eval_active and nn_pred == y_i[0]:
            correct["1-NN"] += 1

        pred_builder = builder.predict(x_i[0])
        if eval_active and pred_builder == y_i[0]:
            correct["oah"] += 1

        for name, model in baselines.items():
            if fitted[name]:
                pred = int(model.predict(x_i)[0])
                if eval_active and pred == y_i[0]:
                    correct[name] += 1
            # Train/update baseline.
            if fitted[name]:
                model.partial_fit(x_i, y_i)
            else:
                model.partial_fit(x_i, y_i, classes=np.array([0, 1]))
                fitted[name] = True

        # Kernel models (RBF kernel, no feature expansion).
        for name, model in kernel_models.items():
            pred = model.predict(x_i[0])
            if eval_active and pred == y_i[0]:
                correct[name] += 1
            model.partial_fit(x_i, y_i)

        # River models (dict-based features).
        x_dict = {0: float(x_i[0, 0]), 1: float(x_i[0, 1])}
        for name, model in river_models.items():
            pred = model.predict_one(x_dict)
            if eval_active and pred is not None and int(pred) == y_i[0]:
                correct[name] += 1
            model.learn_one(x_dict, int(y_i[0]))

        total += 1
        builder.add_point(x_i[0], int(y_i[0]))
        if y_i[0] == 1:
            seen_1 += 1
        else:
            seen_0 += 1
        seen_X.append(x_i[0].copy())
        seen_y.append(int(y_i[0]))
        # Record cumulative accuracies.
        for name in correct:
            history_pre[name].append(correct[name] / eval_total if eval_total > 0 else 0.0)

        # Remaining-set accuracy (on unseen points), evaluated sparsely for speed.
        do_eval = (step in eval_points)
        if not do_eval:
            for name in correct:
                prev = history_rem[name][-1] if history_rem[name] else history_pre[name][-1]
                history_rem[name].append(prev)
        else:
            remaining_idx = order[step + 1 :]
            if len(remaining_idx) == 0:
                for name in correct:
                    prev = history_rem[name][-1] if history_rem[name] else history_pre[name][-1]
                    history_rem[name].append(prev)
            else:
                X_rem = X[remaining_idx]
                y_rem = y[remaining_idx]

                preds = [builder.predict(pt) for pt in X_rem]
                acc = float(np.mean(np.array(preds) == y_rem))
                history_rem["oah"].append(acc)

                maj_pred = 1 if seen_1 > seen_0 else 0
                acc_m = float(np.mean(y_rem == maj_pred))
                history_rem["Majority"].append(acc_m)

                if seen_X:
                    seen_stack = np.vstack(seen_X)
                    y_seen_arr = np.array(seen_y)
                    dists = np.linalg.norm(X_rem[:, None, :] - seen_stack[None, :, :], axis=2)
                    nn_idx = np.argmin(dists, axis=1)
                    nn_preds = y_seen_arr[nn_idx]
                    acc_nn = float(np.mean(nn_preds == y_rem))
                else:
                    acc_nn = float(np.mean(y_rem == 0))
                history_rem["1-NN"].append(acc_nn)

                for name, model in baselines.items():
                    if not fitted[name]:
                        prev = history_rem[name][-1] if history_rem[name] else history_pre[name][-1]
                        history_rem[name].append(prev)
                        continue
                    try:
                        preds_b = model.predict(X_rem)
                        acc_b = float(np.mean(preds_b == y_rem))
                    except Exception:
                        acc_b = history_rem[name][-1] if history_rem[name] else history_pre[name][-1]
                    history_rem[name].append(acc_b)

                for name, model in kernel_models.items():
                    preds_k = [model.predict(pt) for pt in X_rem]
                    acc_k = float(np.mean(np.array(preds_k) == y_rem))
                    history_rem[name].append(acc_k)

                for name, model in river_models.items():
                    preds_r = [model.predict_one({0: float(pt[0]), 1: float(pt[1])}) for pt in X_rem]
                    preds_r = [int(p) if p is not None else 0 for p in preds_r]
                    acc_r = float(np.mean(np.array(preds_r) == y_rem))
                    history_rem[name].append(acc_r)
        if progress_bar is not None:
            progress_bar.update(1)

    if make_plot:
        xs = np.arange(1, total + 1)
        plot_histories(xs, history_pre, history_rem, dataset, plot_path, max_rem_steps=forward_steps)

    acc = {name: (correct[name] / eval_total if eval_total > 0 else 0.0) for name in correct}
    return acc, history_pre, history_rem, len(builder.hyperplanes)


def eval_dataset_2d(
    config: tuple[str, str, int, float, int],
    n_repeats: int,
    make_plots: bool,
    progress: object | None = None,
    progress_lock: object | None = None,
) -> tuple[
    str,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    float,
    float,
    str,
    dict[str, str | int],
]:
    name, dataset_key, n, noise, seed = config
    plot_path = f"scripts/img/compare_baselines_accuracy_{dataset_key}.png"
    acc_runs: list[dict[str, float]] = []
    aulc_runs: list[dict[str, float]] = []
    K_runs: list[int] = []
    for r in range(n_repeats):
        seed_r = seed + r * 17
        acc, _, history_rem, K = run_online_eval(
            n=n,
            noise=noise,
            random_state=seed_r,
            dataset=dataset_key,
            make_plot=(r == 0 and make_plots),
            plot_path=plot_path,
            progress_bar=None,
        )
        if progress is not None and progress_lock is not None:
            with progress_lock:
                progress.value += n
        acc_runs.append(acc)
        aulc_runs.append(compute_aulc(history_rem, max_steps=20))
        K_runs.append(K)
    mean_acc, std_acc = aggregate_results(acc_runs)
    mean_aulc, std_aulc = aggregate_results(aulc_runs)
    k_mean = float(np.mean(K_runs))
    k_std = float(np.std(K_runs))
    Ks = f"{k_mean:.2f}±{k_std:.2f}"
    meta = {
        "n_cols_final": 2,
        "n_rows_total": n,
        "n_rows_used": f"{n:.0f}±0",
        "majority_pct_used": "50.0%±0.0%",
    }
    return name, mean_acc, std_acc, mean_aulc, std_aulc, k_mean, k_std, Ks, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare OAH against online baselines (2D datasets).")
    parser.add_argument("--plot", action="store_true", help="Enable saving accuracy plots.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel workers.")
    args = parser.parse_args()

    dataset_configs = [
        ("xor_noise0.10", "xor", 400, 0.10, 123),
        ("moons_noise0.20", "moons", 400, 0.20, 456),
        ("circles_noise0.08", "circles", 400, 0.08, 654),
        ("gaussians_far", "gauss_far", 400, 0.0, 111),
        ("gaussians_overlap", "gauss_overlap", 400, 0.0, 222),
        ("piecewise_3x3", "piecewise", 900, 0.10, 789),
    ]
    all_results: dict[str, dict[str, float]] = {}
    all_results_std: dict[str, dict[str, float]] = {}
    Ks: dict[str, str] = {}
    K_means: dict[str, float] = {}
    K_stds: dict[str, float] = {}
    metas: dict[str, dict[str, str | int]] = {}
    aulc_results: dict[str, dict[str, float]] = {}
    aulc_results_std: dict[str, dict[str, float]] = {}
    n_repeats = 10
    if args.jobs > 1:
        total_samples = sum(cfg[2] for cfg in dataset_configs) * n_repeats
        manager = mp.Manager()
        progress = manager.Value("i", 0)
        progress_lock = manager.Lock()
        with tqdm(total=total_samples, desc="Samples", leave=False) as bar:
            with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futures = [
                    ex.submit(eval_dataset_2d, cfg, n_repeats, args.plot, progress, progress_lock)
                    for cfg in dataset_configs
                ]
                done = 0
                while done < len(futures):
                    done = sum(1 for f in futures if f.done())
                    with progress_lock:
                        bar.n = progress.value
                    bar.refresh()
                    time.sleep(0.2)
                for fut in futures:
                    name, mean_acc, std_acc, mean_aulc, std_aulc, k_mean, k_std, k_str, meta = fut.result()
                    all_results[name] = mean_acc
                    all_results_std[name] = std_acc
                    aulc_results[name] = mean_aulc
                    aulc_results_std[name] = std_aulc
                    Ks[name] = k_str
                    K_means[name] = k_mean
                    K_stds[name] = k_std
                    metas[name] = meta
                    print(f"\n=== Dataset: {name} ===")
                    print(f"OAH builder K mean={k_str}")
    else:
        for cfg in dataset_configs:
            name, mean_acc, std_acc, mean_aulc, std_aulc, k_mean, k_std, k_str, meta = eval_dataset_2d(
                cfg, n_repeats, args.plot
            )
            all_results[name] = mean_acc
            all_results_std[name] = std_acc
            aulc_results[name] = mean_aulc
            aulc_results_std[name] = std_aulc
            Ks[name] = k_str
            K_means[name] = k_mean
            K_stds[name] = k_std
            metas[name] = meta
            print(f"\n=== Dataset: {name} ===")
            print(f"OAH builder K mean={k_str}")
    dataset_names = [cfg[0] for cfg in dataset_configs]
    print_table_stats("Prequential accuracy (mean±std)", dataset_names, all_results, all_results_std, "acc")
    print_table_stats("ISA@20 (remaining set, mean±std)", dataset_names, aulc_results, aulc_results_std, "ISA@20")
    print_oah_summary([cfg[0] for cfg in dataset_configs], Ks, metas)

    method_labels = {"oah": "OAH (ours)", "OAH": "OAH (ours)"}
    k_mean_results: dict[str, dict[str, float]] = {}
    k_std_results: dict[str, dict[str, float]] = {}
    for d in dataset_names:
        k_mean_results[d] = {"oah": K_means[d]}
        k_std_results[d] = {"oah": K_stds[d]}
    out_path = Path("results/oah_baselines_2d.csv")
    save_results_csv(
        out_path,
        dataset_names,
        [
            ("prequential_accuracy", all_results, all_results_std),
            ("isa@20", aulc_results, aulc_results_std),
            ("K", k_mean_results, k_std_results),
        ],
        method_labels,
    )
    print(f"Saved CSV results to {out_path}")
