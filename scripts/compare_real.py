"""
Compare OAH (Online Additive Hyperplanes) against online baselines on real/high-D datasets.
Currently includes the Colon cancer microarray dataset.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import argparse
import concurrent.futures as cf
import multiprocessing as mp
import sys
import time
from sklearn.linear_model import Perceptron, SGDClassifier
from tqdm import tqdm
from tqdm import trange
from river import linear_model as rv_linear
from river import ensemble as rv_ens
from river import tree as rv_tree

from oah import OnlineAdditiveHyperplanes
from oah.datasets import (
    make_airlines,
    make_colon,
    make_electricity,
    make_higgs_small,
    make_breast_cancer,
    make_sonar,
    make_phishing,
    make_spambase,
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
    random_state: int = 123,
    dataset: str = "colon",
    n_limit: int | None = None,
    make_plot: bool = False,
    plot_path: str = "scripts/img/compare_real_accuracy.png",
    forward_steps: int = 20,
    burn_in_prequential: int = 1,
    progress_bar: tqdm | None = None,
    total_repeats: int | None = None,
) -> tuple[dict[str, float], dict[str, list[float]], dict[str, list[float]], int, dict[str, int], dict[str, float]]:
    if dataset == "colon":
        X, y = make_colon(random_state=random_state)
        meta = {"n_rows": len(X), "n_cols_raw": X.shape[1], "n_numeric_cols": X.shape[1], "n_categorical_cols": 0, "n_cols_final": X.shape[1]}
    elif dataset == "spambase":
        X, y, meta = make_spambase(random_state=random_state, n_limit=n_limit)
    elif dataset == "electricity":
        X, y, meta = make_electricity(random_state=random_state, n_limit=n_limit)
    elif dataset == "airlines":
        X, y, meta = make_airlines(random_state=random_state, n_limit=n_limit)
    elif dataset == "phishing":
        X, y, meta = make_phishing(random_state=random_state, n_limit=n_limit)
    elif dataset == "higgs":
        X, y, meta = make_higgs_small(random_state=random_state, n_limit=n_limit)
    elif dataset == "breast_cancer":
        X, y, meta = make_breast_cancer(random_state=random_state, n_limit=n_limit)
    elif dataset == "sonar":
        X, y, meta = make_sonar(random_state=random_state, n_limit=n_limit)
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(X))
    total_samples = len(order)
    if progress_bar is not None and total_repeats is not None:
        progress_bar.total = total_samples * total_repeats
        progress_bar.refresh()

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
        "OAH": 0,
        "Majority": 0,
        "1-NN": 0,
        **{name: 0 for name in baselines},
        **{name: 0 for name in kernel_models},
        **{name: 0 for name in river_models},
    }
    timings = {name: 0.0 for name in correct}
    history_pre = {name: [] for name in correct}
    history_rem = {name: [] for name in correct}
    total = 0
    eval_total = 0
    seen_0 = 0
    seen_1 = 0

    eval_points = set(np.unique(np.linspace(0, total_samples - 1, num=min(10, total_samples), dtype=int)))
    eval_points.update(range(min(forward_steps, total_samples)))

    majority_pct = float(max(np.mean(y), 1.0 - np.mean(y))) if len(y) else float("nan")
    meta["n_rows_total"] = meta.get("n_rows", len(X))
    meta["n_rows_used"] = len(X)
    meta["majority_pct_used"] = majority_pct

    seen_X: list[np.ndarray] = []
    seen_y: list[int] = []
    for step, idx in enumerate(order):
        x_i = X[idx].reshape(1, -1)
        y_i = np.array([y[idx]])
        eval_active = step >= burn_in_prequential
        if eval_active:
            eval_total += 1

        # Predict before updating (online evaluation).
        start_t = time.perf_counter()
        if seen_1 > seen_0:
            majority_pred = 1
        else:
            majority_pred = 0
        if eval_active and majority_pred == y_i[0]:
            correct["Majority"] += 1
        timings["Majority"] += time.perf_counter() - start_t

        start_t = time.perf_counter()
        if seen_X:
            dists = np.linalg.norm(np.vstack(seen_X) - x_i[0], axis=1)
            nn_idx = int(np.argmin(dists))
            nn_pred = int(seen_y[nn_idx])
        else:
            nn_pred = 0
        if eval_active and nn_pred == y_i[0]:
            correct["1-NN"] += 1
        timings["1-NN"] += time.perf_counter() - start_t

        start_t = time.perf_counter()
        pred_builder = builder.predict(x_i[0])
        if eval_active and pred_builder == y_i[0]:
            correct["OAH"] += 1
        timings["OAH"] += time.perf_counter() - start_t

        for name, model in baselines.items():
            start_t = time.perf_counter()
            if fitted[name]:
                pred = int(model.predict(x_i)[0])
                if eval_active and pred == y_i[0]:
                    correct[name] += 1
            if fitted[name]:
                model.partial_fit(x_i, y_i)
            else:
                model.partial_fit(x_i, y_i, classes=np.array([0, 1]))
                fitted[name] = True
            timings[name] += time.perf_counter() - start_t

        for name, model in kernel_models.items():
            start_t = time.perf_counter()
            pred = model.predict(x_i[0])
            if eval_active and pred == y_i[0]:
                correct[name] += 1
            model.partial_fit(x_i, y_i)
            timings[name] += time.perf_counter() - start_t

        x_dict = {i: float(v) for i, v in enumerate(x_i[0])}
        for name, model in river_models.items():
            start_t = time.perf_counter()
            pred = model.predict_one(x_dict)
            if eval_active and pred is not None and int(pred) == y_i[0]:
                correct[name] += 1
            model.learn_one(x_dict, int(y_i[0]))
            timings[name] += time.perf_counter() - start_t

        total += 1
        start_t = time.perf_counter()
        builder.add_point(x_i[0], int(y_i[0]))
        timings["OAH"] += time.perf_counter() - start_t
        if y_i[0] == 1:
            seen_1 += 1
        else:
            seen_0 += 1
        seen_X.append(x_i[0].copy())
        seen_y.append(int(y_i[0]))
        for name in correct:
            history_pre[name].append(correct[name] / eval_total if eval_total > 0 else 0.0)

        # Remaining-set accuracy (sparse for speed).
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

                start_t = time.perf_counter()
                preds = [builder.predict(pt) for pt in X_rem]
                acc = float(np.mean(np.array(preds) == y_rem))
                history_rem["OAH"].append(acc)
                timings["OAH"] += time.perf_counter() - start_t

                start_t = time.perf_counter()
                maj_pred = 1 if seen_1 > seen_0 else 0
                acc_m = float(np.mean(y_rem == maj_pred))
                history_rem["Majority"].append(acc_m)
                timings["Majority"] += time.perf_counter() - start_t

                start_t = time.perf_counter()
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
                timings["1-NN"] += time.perf_counter() - start_t

                for name, model in baselines.items():
                    start_t = time.perf_counter()
                    if not fitted[name]:
                        prev = history_rem[name][-1] if history_rem[name] else history_pre[name][-1]
                        history_rem[name].append(prev)
                        timings[name] += time.perf_counter() - start_t
                        continue
                    try:
                        preds_b = model.predict(X_rem)
                        acc_b = float(np.mean(preds_b == y_rem))
                    except Exception:
                        acc_b = history_rem[name][-1] if history_rem[name] else history_pre[name][-1]
                    history_rem[name].append(acc_b)
                    timings[name] += time.perf_counter() - start_t

                for name, model in kernel_models.items():
                    start_t = time.perf_counter()
                    preds_k = [model.predict(pt) for pt in X_rem]
                    acc_k = float(np.mean(np.array(preds_k) == y_rem))
                    history_rem[name].append(acc_k)
                    timings[name] += time.perf_counter() - start_t

                for name, model in river_models.items():
                    start_t = time.perf_counter()
                    preds_r = []
                    for pt in X_rem:
                        pt_dict = {i: float(v) for i, v in enumerate(pt)}
                        pred = model.predict_one(pt_dict)
                        preds_r.append(int(pred) if pred is not None else 0)
                    acc_r = float(np.mean(np.array(preds_r) == y_rem))
                    history_rem[name].append(acc_r)
                    timings[name] += time.perf_counter() - start_t
        if progress_bar is not None:
            progress_bar.update(1)

    if make_plot:
        xs = np.arange(1, total + 1)
        plot_histories(xs, history_pre, history_rem, dataset, plot_path, max_rem_steps=forward_steps)

    acc = {name: (correct[name] / eval_total if eval_total > 0 else 0.0) for name in correct}
    return acc, history_pre, history_rem, len(builder.hyperplanes), meta, timings


def eval_dataset_real(
    config: tuple[str, str, int | None, int],
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
    dict[str, float],
]:
    name, dataset_key, n_limit, seed = config
    plot_path = f"scripts/img/compare_real_accuracy_{dataset_key}.png"
    acc_runs: list[dict[str, float]] = []
    aulc_runs: list[dict[str, float]] = []
    K_runs: list[int] = []
    meta: dict[str, str | int] = {}
    used_rows_runs: list[int] = []
    majority_runs: list[float] = []
    total_timings: dict[str, float] = {}
    for r in range(n_repeats):
        seed_r = seed + r * 17
        acc, _, history_rem, K, meta, timings = run_online_eval(
            random_state=seed_r,
            dataset=dataset_key,
            n_limit=n_limit,
            make_plot=(r == 0 and make_plots),
            plot_path=plot_path,
            progress_bar=None,
            total_repeats=n_repeats,
        )
        if progress is not None and progress_lock is not None:
            with progress_lock:
                progress.value += int(meta.get("n_rows_used", 0))
        for model_name, elapsed in timings.items():
            total_timings[model_name] = total_timings.get(model_name, 0.0) + elapsed
        used_rows_runs.append(int(meta.get("n_rows_used", 0)))
        majority_runs.append(float(meta.get("majority_pct_used", float("nan"))))
        acc_runs.append(acc)
        aulc_runs.append(compute_aulc(history_rem, max_steps=20))
        K_runs.append(K)
    mean_acc, std_acc = aggregate_results(acc_runs)
    mean_aulc, std_aulc = aggregate_results(aulc_runs)
    k_mean = float(np.mean(K_runs))
    k_std = float(np.std(K_runs))
    Ks = f"{k_mean:.2f}±{k_std:.2f}"
    majority_mean = float(np.mean(majority_runs)) if majority_runs else float("nan")
    majority_std = float(np.std(majority_runs)) if majority_runs else float("nan")
    used_mean = float(np.mean(used_rows_runs)) if used_rows_runs else float("nan")
    used_std = float(np.std(used_rows_runs)) if used_rows_runs else float("nan")
    meta["n_rows_total"] = int(meta.get("n_rows_total", meta.get("n_rows", 0)))
    meta["n_rows_used"] = f"{used_mean:.0f}±{used_std:.0f}" if np.isfinite(used_mean) else "nan"
    meta["majority_pct_used"] = f"{majority_mean*100:.1f}%±{majority_std*100:.1f}%" if np.isfinite(majority_mean) else "nan"
    return name, mean_acc, std_acc, mean_aulc, std_aulc, k_mean, k_std, Ks, meta, total_timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare OAH against online baselines (real datasets).")
    parser.add_argument("--plot", action="store_true", help="Enable saving accuracy plots.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel workers.")
    args = parser.parse_args()

    dataset_configs = [
        ("colon", "colon", 1000, 321),
        ("spambase", "spambase", 1000, 111),
        ("electricity", "electricity", 1000, 222),
        ("phishing", "phishing", 1000, 444),
        ("breast_cancer", "breast_cancer", 1000, 666),
        ("sonar", "sonar", 1000, 777),
    ]
    all_results: dict[str, dict[str, float]] = {}
    all_results_std: dict[str, dict[str, float]] = {}
    Ks: dict[str, str] = {}
    K_means: dict[str, float] = {}
    K_stds: dict[str, float] = {}
    metas: dict[str, dict[str, str | int]] = {}
    aulc_results: dict[str, dict[str, float]] = {}
    aulc_results_std: dict[str, dict[str, float]] = {}
    total_timings: dict[str, float] = {}
    n_repeats = 10
    if args.jobs > 1:
        total_samples = sum(cfg[2] for cfg in dataset_configs if cfg[2] is not None) * n_repeats
        manager = mp.Manager()
        progress = manager.Value("i", 0)
        progress_lock = manager.Lock()
        with tqdm(total=total_samples, desc="Samples", leave=False) as bar:
            with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futures = [
                    ex.submit(eval_dataset_real, cfg, n_repeats, args.plot, progress, progress_lock)
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
                    name, mean_acc, std_acc, mean_aulc, std_aulc, k_mean, k_std, k_str, meta, timings = fut.result()
                    all_results[name] = mean_acc
                    all_results_std[name] = std_acc
                    aulc_results[name] = mean_aulc
                    aulc_results_std[name] = std_aulc
                    Ks[name] = k_str
                    K_means[name] = k_mean
                    K_stds[name] = k_std
                    metas[name] = meta
                    for model_name, elapsed in timings.items():
                        total_timings[model_name] = total_timings.get(model_name, 0.0) + elapsed
                    print(f"\n=== Dataset: {name} ===")
                    print(f"OAH K mean={k_str}")
                    print(f"Meta: rows={meta.get('n_rows')} cols={meta.get('n_cols_final')}")
    else:
        for cfg in dataset_configs:
            name, mean_acc, std_acc, mean_aulc, std_aulc, k_mean, k_std, k_str, meta, timings = eval_dataset_real(
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
            for model_name, elapsed in timings.items():
                total_timings[model_name] = total_timings.get(model_name, 0.0) + elapsed
            print(f"\n=== Dataset: {name} ===")
            print(f"OAH K mean={k_str}")
            print(f"Meta: rows={meta.get('n_rows')} cols={meta.get('n_cols_final')}")
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
    out_path = Path("results/oah_baselines_real.csv")
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
    if total_timings:
        print("\nTotal runtime by model (seconds, sum over datasets and repeats):")
        for model_name, elapsed in sorted(total_timings.items(), key=lambda item: item[1], reverse=True):
            print(f"{model_name:14s} | {elapsed:8.2f}")
