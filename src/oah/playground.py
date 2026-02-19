"""
Interactive 1-NN "external angle" explorer (3 points, 2 classes).

Launch with:
    poetry run python -m oah.playground
"""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib
import platform
print(f"[playground] platform={platform.platform()}", flush=True)
print(f"[playground] matplotlib backend before import: {matplotlib.get_backend()}", flush=True)
try:
    import PyQt5  # noqa: F401
    print("[playground] PyQt5 is available.", flush=True)
except Exception as exc:
    print(f"[playground] PyQt5 not available: {exc}", flush=True)

from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

matplotlib.use("QtAgg", force=True)  # enforce Qt backend for consistent event handling
import numpy as np


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2D cross-product (z-component)."""
    return float(a[0] * b[1] - a[1] * b[0])


@dataclass
class FractionResult:
    collinear: bool
    theta_rad: float
    alpha_rad: float
    f_b: float
    f_a: float
    note: str


def compute_fractions(a1: np.ndarray, a2: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> FractionResult:
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    b = np.asarray(b, dtype=float)

    d = a2 - a1
    denom = float(np.dot(d, d))
    if denom < eps:
        return FractionResult(True, math.nan, math.nan, 0.5, 0.5, "Degenerate: A1 and A2 overlap.")

    area2 = abs(cross2(d, b - a1))
    collinear = area2 <= eps * denom

    if collinear:
        t = float(np.dot(b - a1, d) / denom)
        between = 0.0 <= t <= 1.0
        f_b = 0.0 if between else 0.5
        note = "Collinear: B between A1 and A2 ⇒ f_B=0" if between else "Collinear: B outside segment ⇒ f_B=1/2"
        return FractionResult(True, math.nan, math.nan, f_b, 1.0 - f_b, note)

    u = a1 - b
    v = a2 - b
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < eps or nv < eps:
        return FractionResult(False, math.nan, math.nan, 0.0, 1.0, "Degenerate: B coincides with A1 or A2.")

    c = clamp(float(np.dot(u, v) / (nu * nv)))
    theta = math.acos(c)
    alpha = math.pi - theta
    f_b = alpha / (2.0 * math.pi)
    return FractionResult(False, theta, alpha, f_b, 1.0 - f_b, "Non-collinear: f_B=(π-θ)/2π")


class DraggableApp(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OAH Playground")
        self.domain_min = -1.0
        self.domain_max = 1.0
        self.mc_samples = 1000
        self.rng = np.random.default_rng(42)
        self.region_cmap = ListedColormap(["#7fb3d5", "#f5b041"])
        self.color_mode = "1NN"
        self.use_second_a = True
        self.use_second_b = True
        self.a1 = np.array([-1.0, -0.5])
        self.a2 = np.array([1.0, -0.5])
        self.b1 = np.array([0.0, 0.8])
        self.b2 = np.array([0.2, 0.2])
        self.color_a = np.array([0.44, 0.69, 0.88])
        self.color_b = np.array([0.96, 0.70, 0.27])
        self.dragging: str | None = None

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(5, 4)))
        self.ax = self.canvas.figure.add_subplot(111)
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.canvas.setFocus()

        toolbar = NavigationToolbar2QT(self.canvas, self)
        central = QtWidgets.QWidget(self)
        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(toolbar)

        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addLayout(content_layout)

        content_layout.addWidget(self.canvas, stretch=3)

        stats_container = QtWidgets.QWidget()
        stats_layout = QtWidgets.QVBoxLayout(stats_container)
        stats_layout.setContentsMargins(10, 0, 0, 0)
        stats_layout.addWidget(QtWidgets.QLabel("Metrics"), alignment=QtCore.Qt.AlignTop)
        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setTextFormat(QtCore.Qt.PlainText)
        self.stats_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.stats_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch(1)
        content_layout.addWidget(stats_container, stretch=1)

        self.setCentralWidget(central)

        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)
        view_menu = menu_bar.addMenu("View")
        self.color_mode_group = QtWidgets.QActionGroup(self)
        self.color_mode_group.setExclusive(True)
        for mode, label, checked in [
            ("1NN", "Color by 1-NN", True),
            ("2NN", "Color by 2-NN", False),
            ("3NN", "Color by 3-NN", False),
        ]:
            action = view_menu.addAction(label)
            action.setCheckable(True)
            action.setData(mode)
            action.setChecked(checked)
            self.color_mode_group.addAction(action)
        self.color_mode_group.triggered.connect(self.on_color_mode_changed)
        view_menu.addSeparator()
        self.toggle_a2_action = view_menu.addAction("Show A2")
        self.toggle_a2_action.setCheckable(True)
        self.toggle_a2_action.setChecked(True)
        self.toggle_a2_action.toggled.connect(self.on_toggle_a2)
        self.toggle_b2_action = view_menu.addAction("Show B2")
        self.toggle_b2_action.setCheckable(True)
        self.toggle_b2_action.setChecked(True)
        self.toggle_b2_action.toggled.connect(self.on_toggle_b2)

        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_xlim(self.domain_min, self.domain_max)
        self.ax.set_ylim(self.domain_min, self.domain_max)
        self.ax.set_axis_off()
        self.canvas.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set_position([0.0, 0.0, 1.0, 1.0])
        self.scatter_a1 = self.ax.scatter(self.a1[0], self.a1[1], c="tab:blue", s=120, picker=8)
        self.scatter_a2 = self.ax.scatter(self.a2[0], self.a2[1], c="tab:blue", s=120, picker=8)
        self.scatter_b1 = self.ax.scatter(self.b1[0], self.b1[1], c="tab:orange", s=140, picker=8, label="B1")
        self.scatter_b2 = self.ax.scatter(self.b2[0], self.b2[1], c="tab:orange", s=140, picker=8, label="B2")

        self.label_a1 = self.ax.text(self.a1[0], self.a1[1], "  A1", va="center")
        self.label_a2 = self.ax.text(self.a2[0], self.a2[1], "  A2", va="center")
        self.label_b1 = self.ax.text(self.b1[0], self.b1[1], "  B1", va="center")
        self.label_b2 = self.ax.text(self.b2[0], self.b2[1], "  B2", va="center")

        (self.line_b1a1,) = self.ax.plot([], [], linewidth=1, color="tab:gray", alpha=0.5)
        (self.line_b1a2,) = self.ax.plot([], [], linewidth=1, color="tab:gray", alpha=0.5)
        (self.line_b2a1,) = self.ax.plot([], [], linewidth=1, color="tab:gray", alpha=0.5)
        (self.line_b2a2,) = self.ax.plot([], [], linewidth=1, color="tab:gray", alpha=0.5)

        self.mc_points = self.rng.uniform(self.domain_min, self.domain_max, size=(self.mc_samples, 2))
        self.mc_scatter = self.ax.scatter(
            self.mc_points[:, 0],
            self.mc_points[:, 1],
            c=np.zeros(self.mc_samples),
            cmap=self.region_cmap,
            vmin=0.0,
            vmax=1.0,
            s=20,
            alpha=0.25,
            edgecolors="none",
        )

        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("key_press_event", self.on_key)

        self.update()

    def clamp_point(self, point: np.ndarray) -> np.ndarray:
        return np.clip(point, self.domain_min, self.domain_max)

    def on_pick(self, event) -> None:
        if event.artist is self.scatter_a1:
            self.dragging = "a1"
        elif event.artist is self.scatter_a2:
            self.dragging = "a2"
        elif event.artist is self.scatter_b1:
            self.dragging = "b1"
        elif event.artist is self.scatter_b2:
            self.dragging = "b2"
        else:
            self.dragging = None

    def on_press(self, event) -> None:
        if event.button != 1:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        pos = np.array([event.xdata, event.ydata], dtype=float)
        thresh = 0.1
        if np.linalg.norm(pos - self.a1) <= thresh:
            self.dragging = "a1"
        elif np.linalg.norm(pos - self.a2) <= thresh:
            self.dragging = "a2"
        elif np.linalg.norm(pos - self.b1) <= thresh:
            self.dragging = "b1"
        elif np.linalg.norm(pos - self.b2) <= thresh:
            self.dragging = "b2"

    def on_motion(self, event) -> None:
        if self.dragging is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        point = np.array([event.xdata, event.ydata], dtype=float)
        point = self.clamp_point(point)
        if self.dragging == "a1":
            self.a1 = point
        elif self.dragging == "a2":
            self.a2 = point
        elif self.dragging == "b1":
            self.b1 = point
        elif self.dragging == "b2":
            self.b2 = point
        self.update()

    def on_release(self, _event) -> None:
        self.dragging = None

    def on_key(self, event) -> None:
        if event.key in {"q", "escape"}:
            self.close()
        elif event.key == "s":
            self.save_snapshot()
        elif event.key == "r":
            self.a1 = np.array([-1.0, -0.5])
            self.a2 = np.array([1.0, -0.5])
            self.b1 = np.array([0.0, 0.8])
            self.b2 = np.array([0.2, 0.2])
            self.update()

    def keyPressEvent(self, event) -> None:  # Qt fallback
        if event.key() in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
            self.close()
        elif event.key() == QtCore.Qt.Key_S:
            self.save_snapshot()
        elif event.key() == QtCore.Qt.Key_R:
            self.a1 = np.array([-1.0, -0.5])
            self.a2 = np.array([1.0, -0.5])
            self.b1 = np.array([0.0, 0.8])
            self.b2 = np.array([0.2, 0.2])
            self.update()
        else:
            super().keyPressEvent(event)

    def save_snapshot(self) -> None:
        output_dir = Path.cwd()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"playground_snapshot_{timestamp}.png"
        self.canvas.figure.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
        print(f"[playground] saved {output_path}", flush=True)

    def on_color_mode_changed(self, action: QtWidgets.QAction) -> None:
        mode = action.data()
        if mode not in {"1NN", "2NN", "3NN"}:
            return
        self.color_mode = mode
        self.update()

    def on_toggle_a2(self, checked: bool) -> None:
        self.use_second_a = checked
        self.update()

    def on_toggle_b2(self, checked: bool) -> None:
        self.use_second_b = checked
        self.update()

    def bisector_segment(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        mid = 0.5 * (p + q)
        diff = q - p
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            return np.vstack([mid, mid])
        perp = np.array([-diff[1], diff[0]]) / norm
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        span = max(x1 - x0, y1 - y0, 1.0) * 2.0
        return np.vstack([mid - perp * span, mid + perp * span])

    def update(self) -> None:
        self.a1 = self.clamp_point(self.a1)
        self.a2 = self.clamp_point(self.a2)
        self.b1 = self.clamp_point(self.b1)
        self.b2 = self.clamp_point(self.b2)

        self.scatter_a1.set_offsets([self.a1])
        self.scatter_a2.set_offsets([self.a2])
        self.scatter_a2.set_visible(self.use_second_a)
        self.scatter_b1.set_offsets([self.b1])
        self.scatter_b2.set_offsets([self.b2])
        self.scatter_b2.set_visible(self.use_second_b)
        self.label_a1.set_position(self.a1)
        self.label_a2.set_position(self.a2)
        self.label_a2.set_visible(self.use_second_a)
        self.label_b1.set_position(self.b1)
        self.label_b2.set_position(self.b2)
        self.label_b2.set_visible(self.use_second_b)

        a2_for_angle = self.a2 if self.use_second_a else self.a1
        result_b1 = compute_fractions(self.a1, a2_for_angle, self.b1)
        result_b2 = compute_fractions(self.a1, a2_for_angle, self.b2) if self.use_second_b else None

        a_points = [self.a1]
        if self.use_second_a:
            a_points.append(self.a2)
        b_points = [self.b1]
        if self.use_second_b:
            b_points.append(self.b2)

        a_stack = np.stack(a_points, axis=0)
        b_stack = np.stack(b_points, axis=0)
        dists_a = np.linalg.norm(self.mc_points[:, None, :] - a_stack[None, :, :], axis=2)
        dists_b = np.linalg.norm(self.mc_points[:, None, :] - b_stack[None, :, :], axis=2)
        d_a = dists_a.min(axis=1)
        d_b = dists_b.min(axis=1)
        mask_b = d_b < d_a

        class_labels = ["A"] * len(a_points) + ["B"] * len(b_points)
        class_arr = np.array(class_labels)
        dist_matrix = np.concatenate([dists_a, dists_b], axis=1)
        total_refs = dist_matrix.shape[1]
        order = np.argsort(dist_matrix, axis=1)
        first_cls = class_arr[order[:, 0]]
        if total_refs >= 2:
            second_cls = class_arr[order[:, 1]]
            two_nn_agree = first_cls == second_cls
        else:
            second_cls = first_cls
            two_nn_agree = np.zeros_like(first_cls, dtype=bool)
        if total_refs >= 3:
            third_cls = class_arr[order[:, 2]]
            top3 = np.stack([first_cls, second_cls, third_cls], axis=1)
            count_b_top3 = np.sum(top3 == "B", axis=1)
            count_a_top3 = 3 - count_b_top3
            three_nn_class = np.where(
                count_b_top3 > count_a_top3,
                "B",
                np.where(count_a_top3 > count_b_top3, "A", first_cls),
            )
        else:
            three_nn_class = first_cls

        f_b_square = float(mask_b.mean())
        f_a_square = 1.0 - f_b_square

        two_nn_class = np.where(two_nn_agree, first_cls, first_cls)
        if self.color_mode == "1NN":
            chosen_cls = first_cls
        elif self.color_mode == "2NN":
            chosen_cls = two_nn_class
        else:
            chosen_cls = three_nn_class
        base_colors = np.empty((self.mc_samples, 3), dtype=float)
        base_colors[chosen_cls == "B"] = self.color_b
        base_colors[chosen_cls != "B"] = self.color_a
        colors = base_colors.copy()
        dark_mask = two_nn_agree
        light_mask = ~dark_mask
        colors[dark_mask] = np.clip(base_colors[dark_mask] * 0.65, 0.0, 1.0)
        colors[light_mask] = np.clip(base_colors[light_mask] + (1.0 - base_colors[light_mask]) * 0.35, 0.0, 1.0)
        self.mc_scatter.set_facecolors(colors)
        self.mc_scatter.set_edgecolors(colors)
        self.mc_scatter.set_array(None)

        def format_result(label: str, result) -> str:
            if result.collinear:
                return (
                    f"{label}: {result.note}\n"
                    f"f_{label}^∞={result.f_b*100.0:.2f}% | f_A^∞={(result.f_a)*100.0:.2f}%"
                )
            theta_deg = result.theta_rad * 180.0 / math.pi
            alpha_deg = result.alpha_rad * 180.0 / math.pi
            return (
                f"{label}: θ={theta_deg:.2f}°, α={alpha_deg:.2f}°\n"
                f"f_{label}^∞={result.f_b*100.0:.2f}% | f_A^∞={result.f_a*100.0:.2f}%"
            )

        info = format_result("B1", result_b1)
        if self.use_second_b and result_b2 is not None:
            info += "\n\n" + format_result("B2", result_b2)
        else:
            info += "\n\nB2 hidden (enable via View → Show B2)"
        if not self.use_second_a:
            info += "\n(A2 hidden; angles use only A1.)"
        info += f"\n\nMonte Carlo (N={self.mc_samples})\n"
        info += f"f_B^□={f_b_square*100.0:.2f}% | f_A^□={f_a_square*100.0:.2f}%"
        mode_label = {"1NN": "1-NN", "2NN": "2-NN", "3NN": "3-NN"}[self.color_mode]
        info += f"\nColoring mode: {mode_label}"
        info += "\n(2-NN consensus shown as darker dots)"
        self.stats_label.setText(info)

        seg1 = self.bisector_segment(self.b1, self.a1)
        self.line_b1a1.set_data(seg1[:, 0], seg1[:, 1])
        if self.use_second_a:
            seg2 = self.bisector_segment(self.b1, self.a2)
            self.line_b1a2.set_data(seg2[:, 0], seg2[:, 1])
        else:
            self.line_b1a2.set_data([], [])
        if self.use_second_b:
            seg3 = self.bisector_segment(self.b2, self.a1)
            self.line_b2a1.set_data(seg3[:, 0], seg3[:, 1])
            if self.use_second_a:
                seg4 = self.bisector_segment(self.b2, self.a2)
                self.line_b2a2.set_data(seg4[:, 0], seg4[:, 1])
            else:
                self.line_b2a2.set_data([], [])
        else:
            self.line_b2a1.set_data([], [])
            self.line_b2a2.set_data([], [])

        self.canvas.draw_idle()


def main() -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    window = DraggableApp()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
