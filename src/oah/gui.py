"""
Simple PyQt5 GUI to step through the OAH (ours) OnlineAdditiveHyperplanes on small 2D datasets.

Controls:
- Next Point: pobiera kolejny punkt z permutacji i aktualizuje model.
- Wybór datasetu (przyciski) i opcja „Show full dataset”.

Widok:
- Scatter punktów (zobaczone, ostatni punkt wyróżniony), hiper­płaszczyzny i tło z predykcją.
- Wykres accuracy (decided samples) train/test w czasie.
- Panel tekstowy z krótkim podsumowaniem.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

from oah import OnlineAdditiveHyperplanes
from oah.datasets import (
    make_circles,
    make_gaussians_far,
    make_gaussians_overlap,
    make_moons,
    make_noisy_xor,
    make_piecewise,
)


def _scale_to_minus_one_one(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    span = np.where((maxs - mins) > 1e-12, (maxs - mins), 1.0)
    Z = 2.0 * (X - mins) / span - 1.0
    return Z


class OnlineCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class AccCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.fig = Figure(figsize=(4, 2.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class OnlineGui(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OAH (ours) Online Hyperplane Builder")
        self.dataset_name = "xor"
        self.dataset_seed = 123
        self.plot_bounds: tuple[float, float, float, float]
        self._reset_data()
        self.builder = self._new_builder()
        self.steps: list[int] = []
        self.train_acc: list[float] = []
        self.test_acc: list[float] = []
        self.logs: list[str] = []
        self.show_full_dataset = False

        self._init_ui()
        self._redraw()

    # Paper-oriented palette: colorblind-friendly and readable in print.
    C_CLASS0 = "#0072B2"
    C_CLASS1 = "#D55E00"
    C_HPLANE = "#4D4D4D"
    C_GRID = "#BDBDBD"
    C_LAST = "#000000"
    CMAP_BG = ListedColormap(["#BDD7EE", "#F4C7B5"])

    def _init_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        left_panel = QtWidgets.QVBoxLayout()
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_next = QtWidgets.QPushButton("Next Point")
        self.btn_next.clicked.connect(self.on_next)
        btn_layout.addWidget(self.btn_next)
        self.btn_end = QtWidgets.QPushButton("Go to End")
        self.btn_end.clicked.connect(self.on_go_to_end)
        btn_layout.addWidget(self.btn_end)
        self.status_label = QtWidgets.QLabel("Ready")
        btn_layout.addWidget(self.status_label)
        left_panel.addLayout(btn_layout)

        ds_layout_outer = QtWidgets.QVBoxLayout()
        self.ds_group = QtWidgets.QButtonGroup(self)
        self.ds_group.setExclusive(True)
        rows = [
            ("xor", "moons", "circles"),
            ("piecewise", "gauss_far", "gauss_overlap"),
        ]
        self.ds_buttons: dict[str, QtWidgets.QPushButton] = {}
        for row_names in rows:
            row_layout = QtWidgets.QHBoxLayout()
            for name in row_names:
                btn = QtWidgets.QPushButton(name.title())
                btn.setCheckable(True)
                btn.clicked.connect(lambda _, n=name: self.on_dataset_change(n))
                self.ds_group.addButton(btn)
                self.ds_buttons[name] = btn
                row_layout.addWidget(btn)
            ds_layout_outer.addLayout(row_layout)

        self.chk_full = QtWidgets.QCheckBox("Show full dataset")
        self.chk_full.stateChanged.connect(self.on_toggle_full)
        ds_layout_outer.addWidget(self.chk_full)
        left_panel.addLayout(ds_layout_outer)
        if self.dataset_name in self.ds_buttons:
            self.ds_buttons[self.dataset_name].setChecked(True)

        self.canvas = OnlineCanvas()
        left_panel.addWidget(self.canvas)

        layout.addLayout(left_panel, stretch=3)

        right_panel = QtWidgets.QVBoxLayout()
        self.acc_canvas = AccCanvas()
        right_panel.addWidget(self.acc_canvas)
        self.btn_save_main = QtWidgets.QPushButton("Save main plot")
        self.btn_save_main.clicked.connect(self.on_save_main_plot)
        right_panel.addWidget(self.btn_save_main)

        right_panel.addWidget(QtWidgets.QLabel("Summary"))
        self.text = QtWidgets.QTextEdit()
        self.text.setReadOnly(True)
        right_panel.addWidget(self.text)
        layout.addLayout(right_panel, stretch=2)

    def _new_builder(self) -> OnlineAdditiveHyperplanes:
        return OnlineAdditiveHyperplanes(
            random_state=self.dataset_seed,
        )

    def _reset_data(self) -> None:
        # Match compare_2d.py exactly: same dataset params and first seed.
        if self.dataset_name == "xor":
            self.dataset_seed = 123
            X, y = make_noisy_xor(n=400, noise=0.10, random_state=self.dataset_seed)
        elif self.dataset_name == "moons":
            self.dataset_seed = 456
            X, y = make_moons(n=400, noise=0.20, random_state=self.dataset_seed)
        elif self.dataset_name == "circles":
            self.dataset_seed = 654
            X, y = make_circles(n=400, noise=0.08, random_state=self.dataset_seed)
        elif self.dataset_name == "piecewise":
            self.dataset_seed = 789
            X, y = make_piecewise(n=900, random_state=self.dataset_seed)
        elif self.dataset_name == "gauss_far":
            self.dataset_seed = 111
            X, y = make_gaussians_far(n=400, random_state=self.dataset_seed)
        elif self.dataset_name == "gauss_overlap":
            self.dataset_seed = 222
            X, y = make_gaussians_overlap(n=400, random_state=self.dataset_seed)
        else:
            raise ValueError(f"Unknown dataset '{self.dataset_name}'")
        self.X = _scale_to_minus_one_one(np.asarray(X, dtype=float))
        self.y = y.astype(int)
        x_min, y_min = np.min(self.X, axis=0)
        x_max, y_max = np.max(self.X, axis=0)
        dx = max(1e-9, x_max - x_min)
        dy = max(1e-9, y_max - y_min)
        d = max(dx, dy)
        margin = 0.08 * d
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        half = 0.5 * d + margin
        self.plot_bounds = (cx - half, cx + half, cy - half, cy + half)

        rng = np.random.default_rng(self.dataset_seed)
        self.order = rng.permutation(len(self.X))
        self.ptr = 0
        self.seen_indices: list[int] = []

    def on_next(self) -> None:
        if self.ptr >= len(self.order):
            self.status_label.setText("No more points.")
            self.btn_next.setEnabled(False)
            self.btn_end.setEnabled(False)
            return
        idx = int(self.order[self.ptr])
        info = self.builder.add_point(self.X[idx], int(self.y[idx]))
        self.seen_indices.append(idx)
        self.ptr += 1
        reason = info.get("reason", "")
        K = len(self.builder.hyperplanes)
        self.status_label.setText(f"Step {self.ptr}/{len(self.order)} | K={K} | {reason}")
        if self.ptr >= len(self.order):
            self.btn_next.setEnabled(False)
            self.btn_end.setEnabled(False)
        self._redraw(last_idx=idx, info=info)

    def on_go_to_end(self) -> None:
        if self.ptr >= len(self.order):
            self.status_label.setText("No more points.")
            self.btn_next.setEnabled(False)
            self.btn_end.setEnabled(False)
            return
        last_info = None
        last_idx = None
        while self.ptr < len(self.order):
            idx = int(self.order[self.ptr])
            last_info = self.builder.add_point(self.X[idx], int(self.y[idx]))
            self.seen_indices.append(idx)
            self.ptr += 1
            last_idx = idx
        K = len(self.builder.hyperplanes)
        reason = last_info.get("reason", "") if last_info else ""
        self.status_label.setText(f"Step {self.ptr}/{len(self.order)} | K={K} | {reason}")
        self.btn_next.setEnabled(False)
        self.btn_end.setEnabled(False)
        self._redraw(last_idx=last_idx, info=last_info)

    def on_dataset_change(self, name: str) -> None:
        self.dataset_name = name
        for btn_name, btn in self.ds_buttons.items():
            btn.setChecked(btn_name == name)
        self._reset_data()
        self.builder = self._new_builder()
        self.steps.clear()
        self.train_acc.clear()
        self.test_acc.clear()
        self.logs.clear()
        self.status_label.setText(f"Dataset: {name}")
        self.btn_next.setEnabled(True)
        self.btn_end.setEnabled(True)
        self._redraw()

    def on_toggle_full(self, state: int) -> None:
        self.show_full_dataset = bool(state)
        self._redraw()

    def _draw_background(self, ax) -> None:
        if not self.builder.cells:
            return
        xmin, xmax, ymin, ymax = self.plot_bounds
        grid_res = 200
        xs = np.linspace(xmin, xmax, grid_res)
        ys = np.linspace(ymin, ymax, grid_res)
        coords = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        preds = []
        for pt in coords:
            pred = self.builder.predict(pt)
            preds.append(np.nan if pred is None else pred)
        grid = np.array(preds, dtype=float).reshape(grid_res, grid_res)
        im = ax.imshow(
            grid,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap=self.CMAP_BG,
            vmin=0,
            vmax=1,
            alpha=0.20,
            interpolation="nearest",
        )
        im.cmap.set_bad(alpha=0.0)

    def _metrics(self, indices: list[int]) -> Optional[dict]:
        if not indices:
            return None
        X = self.X[indices]
        y = self.y[indices]
        preds = [self.builder.predict(pt) for pt in X]
        decided_mask = [p is not None for p in preds]
        n_total = len(preds)
        n_decided = int(np.count_nonzero(decided_mask))
        if n_decided == 0:
            acc_decided = np.nan
        else:
            acc_decided = float(np.mean(np.array(preds)[decided_mask] == y[decided_mask]))
        reject_rate = 1.0 - n_decided / n_total
        return {"n_total": n_total, "n_decided": n_decided, "acc_decided": acc_decided, "reject_rate": reject_rate}

    def _update_curves(self, step: int, train_metrics: Optional[dict], test_metrics: Optional[dict]) -> None:
        if step == 0:
            return
        self.steps.append(step)

        def acc_val(m: Optional[dict]) -> float:
            if m is None or m["acc_decided"] is None or np.isnan(m["acc_decided"]):
                return float("nan")
            return float(m["acc_decided"])

        self.train_acc.append(acc_val(train_metrics))
        self.test_acc.append(acc_val(test_metrics))

    def _draw_accuracy(self) -> None:
        ax_acc = self.acc_canvas.ax
        ax_acc.clear()
        ax_acc.set_title("Accuracy over steps (OAH (ours), decided samples)")
        ax_acc.set_xlabel("Step")
        ax_acc.set_ylabel("Accuracy")
        if self.steps:
            ax_acc.plot(self.steps, self.train_acc, label="Train", color="#1f77b4")
            ax_acc.plot(self.steps, self.test_acc, label="Test", color="#ff7f0e")
            ax_acc.set_xlim(1, max(self.steps))
        else:
            ax_acc.set_xlim(1, 1)
        ax_acc.set_ylim(0.0, 1.05)
        if self.steps:
            ax_acc.legend(loc="lower right")
        ax_acc.grid(True, alpha=0.3)
        self.acc_canvas.draw_idle()

    def on_save_main_plot(self) -> None:
        title_name = self.dataset_name.replace("_", " ").title()
        default_name = f"oah_plot_{self.dataset_name}.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            f"Save main plot ({title_name})",
            default_name,
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        self.canvas.fig.savefig(path, dpi=300, bbox_inches="tight")

    def _redraw(self, last_idx: Optional[int] = None, info: Optional[dict] = None) -> None:
        ax = self.canvas.ax
        ax.clear()
        xmin, xmax, ymin, ymax = self.plot_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        title_name = self.dataset_name.replace("_", " ").title()
        ax.set_title(title_name)
        ax.grid(True, color=self.C_GRID, alpha=0.35, linewidth=0.8)

        self._draw_background(ax)

        if self.show_full_dataset:
            mask_pos_all = self.y == 1
            mask_neg_all = self.y == 0
            if np.any(mask_neg_all):
                ax.scatter(self.X[mask_neg_all, 0], self.X[mask_neg_all, 1], c=self.C_CLASS0, s=8, alpha=0.20, edgecolors="none")
            if np.any(mask_pos_all):
                ax.scatter(self.X[mask_pos_all, 0], self.X[mask_pos_all, 1], c=self.C_CLASS1, s=8, alpha=0.20, edgecolors="none")

        if self.seen_indices:
            seen_pts = self.X[self.seen_indices]
            seen_labels = self.y[self.seen_indices]
            mask_pos = seen_labels == 1
            mask_neg = seen_labels == 0
            if np.any(mask_neg):
                ax.scatter(
                    seen_pts[mask_neg, 0],
                    seen_pts[mask_neg, 1],
                    c=self.C_CLASS0,
                    label="class 0",
                    s=28,
                    edgecolors="k",
                    linewidths=0.3,
                )
            if np.any(mask_pos):
                ax.scatter(
                    seen_pts[mask_pos, 0],
                    seen_pts[mask_pos, 1],
                    c=self.C_CLASS1,
                    label="class 1",
                    s=28,
                    edgecolors="k",
                    linewidths=0.3,
                )
            if last_idx is not None and self.ptr < len(self.order):
                last_pt = self.X[last_idx]
                ax.scatter(last_pt[0], last_pt[1], facecolors="none", edgecolors=self.C_LAST, s=120, linewidths=1.2, label="last")

        for hp in self.builder.hyperplanes:
            n = hp.normal
            if len(n) != 2:
                continue
            if abs(n[1]) < 1e-8:
                x_line = hp.bias / (n[0] + 1e-12)
                ax.plot([x_line, x_line], [ymin, ymax], linestyle="--", color=self.C_HPLANE, alpha=0.9, linewidth=1.2)
            else:
                xs = np.linspace(xmin, xmax, 200)
                ys = (hp.bias - n[0] * xs) / (n[1] + 1e-12)
                ax.plot(xs, ys, linestyle="--", color=self.C_HPLANE, alpha=0.9, linewidth=1.2)

        # Accuracy plot
        train_metrics = self._metrics(self.seen_indices)
        remaining = [int(i) for i in self.order[self.ptr :]]
        test_metrics = self._metrics(remaining)
        step_num = len(self.seen_indices)
        self._update_curves(step_num, train_metrics, test_metrics)
        self._draw_accuracy()

        self.canvas.draw_idle()
        self._update_text(info)

    def _update_text(self, info: Optional[dict]) -> None:
        train_metrics = self._metrics(self.seen_indices)
        remaining = [int(i) for i in self.order[self.ptr :]]
        test_metrics = self._metrics(remaining)
        lines = [
            f"Seen points: {len(self.seen_indices)}/{len(self.order)} ({self.dataset_name})",
        ]
        if info:
            lines.append(f"Last reason: {info.get('reason', '')}")
            lines.append(f"Δ (cell): {info.get('delta', 0):.3f}")
            lines.append(f"Imp. gain: {info.get('improvement', 0):.3f}")

        def fmt_metrics(title: str, m: Optional[dict]) -> str:
            if m is None:
                return f"{title}: n=0"
            acc = f"{m['acc_decided']:.3f}" if not np.isnan(m["acc_decided"]) else "nan"
            return (
                f"{title}: n={m['n_total']} decided={m['n_decided']} "
                f"acc(decided)={acc} reject_rate={m['reject_rate']:.3f}"
            )

        lines.append(fmt_metrics("Train", train_metrics))
        lines.append(fmt_metrics("Test", test_metrics))
        lines.append("")
        self.text.setPlainText("\n".join(lines))


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    gui = OnlineGui()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
