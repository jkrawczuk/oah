"""
Online hyperplane builder implementing the OAH rules.

The builder maintains:
- a memory of all seen points (X, y),
- a set of hyperplanes H = {(n_k, b_k)},
- per-cell supports keyed by the pattern bitmask of hyperplane outputs.

Growth (GROW) happens only when a cell is conflictual or uncertain; a new
hyperplane is oriented by a top-k opposite-class point and its bias is chosen
by the best 1D threshold on local projections (Gini impurity). Redundant
hyperplanes are filtered by angle/bias separation, and minimum impurity drop is
required before accepting a split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math

import numpy as np


@dataclass
class Hyperplane:
    normal: np.ndarray
    bias: float
    seed_1: Optional[int] = None  # index of a class-1 point used to orient this H
    seed_0: Optional[int] = None  # index of a class-0 point used to orient this H
    last_two_sided_step: int = 0  # last step when both sides had points


@dataclass
class CellStats:
    support_1: List[int] = field(default_factory=list)
    support_0: List[int] = field(default_factory=list)
    conflict_count: int = 0
    seed_1: List[int] = field(default_factory=list)  # provenance from hyperplane seeds (class 1)
    seed_0: List[int] = field(default_factory=list)  # provenance from hyperplane seeds (class 0)

    @property
    def N_1(self) -> int:
        return len(self.support_1)

    @property
    def N_0(self) -> int:
        return len(self.support_0)

    @property
    def N(self) -> int:
        return self.N_1 + self.N_0

    @property
    def delta(self) -> float:
        if self.N == 0:
            return 0.0
        return abs(self.N_1 - self.N_0) / self.N

    @property
    def majority(self) -> Optional[int]:
        if self.N == 0:
            return None
        return 1 if self.N_1 >= self.N_0 else 0


class OnlineAdditiveHyperplanes:
    """
    Online binary classifier following the OAH rules.

    Labels are normalized to {0,1} internally. Exposes `add_point(x, y)` for
    online growth and `predict(x)` for inference.
    """

    def __init__(
        self,
        grow_patience: Optional[int] = None,
        k_opposite: int = 5,
        max_K: int = 100,
        min_impurity_drop: float = 0.05,
        split_criterion: str = "gini",
        redundant_cos_thresh: float = 0.99,
        redundant_bias_frac: float = 0.02,
        random_state: Optional[int] = None,
        confidence_threshold: float = 0.95,
        anchor_strategy: str = "nearest",
    ) -> None:
        if grow_patience is not None and grow_patience <= 0:
            raise ValueError("grow_patience must be positive.")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0, 1].")
        self.grow_patience = grow_patience
        split_criterion = split_criterion.lower().strip()
        if split_criterion not in {
            "gini",
            "gini_unweighted",
            "entropy",
            "entropy_unweighted",
            "misclass",
            "misclass_unweighted",
            "midpoint",
            "max_margin",
        }:
            raise ValueError(f"Unsupported split_criterion '{split_criterion}'")
        self.split_criterion = split_criterion
        self.k_opposite = k_opposite
        self.max_K = max_K
        self.min_impurity_drop = min_impurity_drop
        self.redundant_cos_thresh = redundant_cos_thresh
        self.redundant_bias_frac = redundant_bias_frac
        self.rng = np.random.default_rng(random_state)
        self.confidence_threshold = confidence_threshold
        anchor_strategy = anchor_strategy.lower().strip()
        if anchor_strategy not in {"nearest", "farthest", "centroid"}:
            raise ValueError(f"Unsupported anchor_strategy '{anchor_strategy}'")
        self.anchor_strategy = anchor_strategy
        self.n1_total = 0
        self.n0_total = 0
        self._dynamic_grow = grow_patience is None
        self.points: List[np.ndarray] = []
        self.labels: List[int] = []
        self.hyperplanes: List[Hyperplane] = []
        self.cells: Dict[int, CellStats] = {}
        self.step: int = 0
        self.pruned_count: int = 0  # retained only for backward compatibility; no pruning now

    # ------------------------------------------------------------------ #
    def add_point(self, x: Sequence[float], y: int | float | str) -> Dict[str, object]:
        """Add a point online; may trigger growth. Returns diagnostics."""
        self.step += 1
        label = self._normalize_label(y)
        x_arr = np.asarray(x, dtype=float)
        idx = len(self.points)
        self.points.append(x_arr)
        self.labels.append(label)
        if label == 1:
            self.n1_total += 1
        else:
            self.n0_total += 1

        pattern = self._pattern(x_arr)
        cell = self.cells.setdefault(pattern, CellStats())
        if label == 1:
            cell.support_1.append(idx)
        else:
            cell.support_0.append(idx)

        majority = cell.majority
        confidence = self._cell_confidence(cell)
        certain = confidence >= self.confidence_threshold
        majority_match = majority is not None and majority == label
        interval_now = self._current_patience()
        info: Dict[str, object] = {
            "pattern": pattern,
            "cell_size": cell.N,
            "delta": cell.delta,
            "confidence": confidence,
            "grew": False,
            "accepted": False,
            "redundant": False,
            "improvement": 0.0,
            "reason": "",
            "K": len(self.hyperplanes),
            "grow_patience": interval_now,
        }

        if certain and majority_match:
            info["reason"] = "stable_cell"
            return info

        cell.conflict_count += 1
        if cell.conflict_count < interval_now:
            info["reason"] = "patience"
            return info
        if len(self.hyperplanes) >= self.max_K:
            info["reason"] = "max_K_reached"
            return info

        grew = self._grow(pattern, idx, label, info)
        info["grew"] = grew
        return info

    def _current_patience(self) -> int:
        """
        Return the effective growth interval.

        If grow_patience is None, use a Fibonacci-like schedule based on current K
        (number of hyperplanes), changing every two Fibonacci numbers:
            K<=3   -> 1   (covers 1,2,3)
            4-8    -> 2   (covers 5,8)
            9-21   -> 3   (covers 13,21)
            >=22   -> 4
        """
        if not self._dynamic_grow:
            return int(self.grow_patience)  # type: ignore[arg-type]
        K = len(self.hyperplanes)
        if K <= 3:
            return 1
        if K <= 8:
            return 2
        if K <= 21:
            return 3
        return 4

    def predict(self, x: Sequence[float]) -> int:
        """Predict label (0/1)."""
        proba = self.predict_proba(x)
        _, p1 = proba
        return 1 if p1 >= 0.5 else 0

    def predict_proba(self, x: Sequence[float]) -> Tuple[float, float]:
        """
        Predict class probabilities (p0, p1).

        Heuristic probabilities: dane komórki, a jeśli ich brak – głosy hiper­płaszczyzn (seedy po stronach bitów) lub 0.5/0.5.
        """
        x_arr = np.asarray(x, dtype=float)
        pattern = self._pattern(x_arr)
        cell = self.cells.get(pattern)

        if cell is not None and cell.N_0 != cell.N_1:
            # Non-tied cell with data: use cell stats.
            p1 = cell.N_1 / cell.N
            p0 = 1.0 - p1
        else:
            # No data in cell: fallback to seed votes or uniform if no seeds.
            votes_1, votes_0 = self._seed_votes(pattern)
            total_votes = votes_1 + votes_0
            if total_votes == 0:
                return (0.5, 0.5)
            p1 = votes_1 / total_votes
            p0 = 1.0 - p1
        
        return (p0, p1)


    # ------------------------------------------------------------------ #
    def _pattern(self, x: np.ndarray) -> int:
        bits = 0
        for i, hp in enumerate(self.hyperplanes):
            val = 1 if np.dot(hp.normal, x) >= hp.bias else 0
            bits |= val << i
        return bits

    def _seed_votes(self, pattern: int) -> Tuple[int, int]:
        votes_1 = 0
        votes_0 = 0
        for i, hp in enumerate(self.hyperplanes):
            bit = (pattern >> i) & 1
            if bit == 1:
                if hp.seed_1 is None:
                    raise ValueError(f"Missing seed_1 for hyperplane {i}")
                votes_1 += 1
            else:
                if hp.seed_0 is None:
                    raise ValueError(f"Missing seed_0 for hyperplane {i}")
                votes_0 += 1
        return votes_1, votes_0

    def _global_prior(self) -> float:
        if not self.labels:
            return 0.5
        return float(np.mean(self.labels))

    def _cell_confidence(self, cell: CellStats) -> float:
        """Confidence in [0,1] via z-score of majority vs global prior."""
        if cell.N == 0:
            return 0.0
        n1, n0 = cell.N_1, cell.N_0
        p0 = self._global_prior()
        p_hat = max(n1, n0) / cell.N
        # Two-sided test vs prior p0.
        denom = cell.N * p0 * (1.0 - p0)
        if denom <= 0.0:
            return 1.0
        z = abs(p_hat - p0) / math.sqrt(1.0 / (4 * cell.N) + p0 * (1 - p0) / cell.N)
        # Convert z to confidence ~ (1 - p-value) for two-sided test.
        from math import erf, sqrt

        p_value = max(0.0, min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2))))))
        return 1.0 - p_value

    def _class_weights(self) -> Tuple[float, float]:
        # Inverse-frequency weights to handle imbalance; eps to avoid div by zero.
        eps = 1e-9
        w1 = 1.0 / max(eps, self.n1_total)
        w0 = 1.0 / max(eps, self.n0_total)
        return w1, w0

    def _grow(self, pattern: int, idx: int, label: int, info: Dict[str, object]) -> bool:
        cell = self.cells.get(pattern)
        if cell is None:
            info["reason"] = "missing_cell"
            return False
        if cell.N_1 == 0 or cell.N_0 == 0:
            info["reason"] = "single_class_cell"
            return False

        anchor = self.points[idx]
        opposite_indices = cell.support_0 if label == 1 else cell.support_1
        if len(opposite_indices) == 0:
            info["reason"] = "no_opposite_in_cell"
            return False

        opp_points = np.stack([self.points[i] for i in opposite_indices], axis=0)
        if self.anchor_strategy == "centroid":
            b_point = np.mean(opp_points, axis=0)
            dists = np.linalg.norm(opp_points - b_point, axis=1)
            chosen_idx = int(np.argmin(dists))
        else:
            dists = np.linalg.norm(opp_points - anchor, axis=1)
            order = np.argsort(dists)
            if self.anchor_strategy == "farthest":
                top = order[-min(self.k_opposite, len(order)) :]
            else:
                top = order[: min(self.k_opposite, len(order))]
            chosen_idx = int(self.rng.choice(top))
            b_point = opp_points[chosen_idx]
        b_idx_global = opposite_indices[chosen_idx]

        n_vec = b_point - anchor
        norm = np.linalg.norm(n_vec)
        if norm == 0:
            info["reason"] = "degenerate_pair"
            return False
        n_vec = n_vec / norm

        S_indices = cell.support_1 + cell.support_0
        S_points = np.stack([self.points[i] for i in S_indices], axis=0)
        S_labels = np.array([self.labels[i] for i in S_indices], dtype=int)

        bias, improvement = self._best_bias(n_vec, S_points, S_labels)
        info["improvement"] = improvement
        if improvement < self.min_impurity_drop:
            info["reason"] = "low_impurity_gain"
            return False

        seed_1_idx = idx if label == 1 else b_idx_global
        seed_0_idx = idx if label == 0 else b_idx_global
        pos_point = self.points[seed_1_idx]
        if np.dot(n_vec, pos_point) < bias:
            # Flip orientation so bit=1 corresponds to the class-1 seed side.
            n_vec = -n_vec
            bias = -bias

        redundant = self._is_redundant(n_vec, bias)
        info["redundant"] = redundant
        if redundant:
            info["reason"] = "redundant"
            return False

        # Track seeds (class-specific points that oriented this hyperplane).
        self.hyperplanes.append(
            Hyperplane(
                normal=n_vec,
                bias=bias,
                seed_1=seed_1_idx,
                seed_0=seed_0_idx,
                last_two_sided_step=self.step,
            )
        )
        info["accepted"] = True
        info["reason"] = "grown"
        self._rebuild_cells(reset_conflicts=True)
        return True

    def _best_bias(self, n_vec: np.ndarray, points: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """
        Find the bias minimizing the selected impurity along projections n^T x.
        Returns (bias, improvement).
        """
        w1, w0 = self._class_weights()
        projections = points @ n_vec
        order = np.argsort(projections)
        proj_sorted = projections[order]
        labels_sorted = labels[order]

        total = len(labels_sorted)
        n_pos = int(np.count_nonzero(labels_sorted == 1))
        n_neg = total - n_pos
        impurity_before = self._criterion_impurity(n_pos, n_neg, w1, w0, self.split_criterion)

        # Cannot split if only one unique projection or single class.
        unique_proj = np.unique(proj_sorted)
        if len(unique_proj) == 1 or impurity_before == 0.0:
            return float(unique_proj[0]), 0.0

        best_bias = float(unique_proj[0])
        best_impurity = impurity_before

        if self.split_criterion in {"midpoint", "max_margin"}:
            if self.split_criterion == "midpoint":
                pos_proj = proj_sorted[labels_sorted == 1]
                neg_proj = proj_sorted[labels_sorted == 0]
                if len(pos_proj) == 0 or len(neg_proj) == 0:
                    return float(unique_proj[0]), 0.0
                best_bias = float(0.5 * (np.mean(pos_proj) + np.mean(neg_proj)))
            else:
                max_gap = -1.0
                best_bias = float(unique_proj[0])
                for i in range(total - 1):
                    if labels_sorted[i] == labels_sorted[i + 1]:
                        continue
                    gap = proj_sorted[i + 1] - proj_sorted[i]
                    if gap > max_gap:
                        max_gap = gap
                        best_bias = float(0.5 * (proj_sorted[i] + proj_sorted[i + 1]))
                if max_gap <= 0.0:
                    return float(unique_proj[0]), 0.0
            left_mask = projections <= best_bias
            left_pos = int(np.count_nonzero(labels[left_mask] == 1))
            left_neg = int(np.count_nonzero(labels[left_mask] == 0))
            right_pos = n_pos - left_pos
            right_neg = n_neg - left_neg
            left_total = left_pos + left_neg
            right_total = right_pos + right_neg
            if left_total == 0 or right_total == 0:
                return float(best_bias), 0.0
            imp_left = self._criterion_impurity(left_pos, left_neg, w1, w0, self.split_criterion)
            imp_right = self._criterion_impurity(right_pos, right_neg, w1, w0, self.split_criterion)
            weighted_imp = (left_total / total) * imp_left + (right_total / total) * imp_right
            improvement = impurity_before - weighted_imp
            return float(best_bias), float(improvement)

        left_pos = 0
        left_neg = 0
        for i in range(total - 1):
            if labels_sorted[i] == 1:
                left_pos += 1
            else:
                left_neg += 1
            if proj_sorted[i] == proj_sorted[i + 1]:
                continue
            right_pos = n_pos - left_pos
            right_neg = n_neg - left_neg
            left_total = left_pos + left_neg
            right_total = right_pos + right_neg
            if left_total == 0 or right_total == 0:
                continue
            imp_left = self._criterion_impurity(left_pos, left_neg, w1, w0, self.split_criterion)
            imp_right = self._criterion_impurity(right_pos, right_neg, w1, w0, self.split_criterion)
            weighted_imp = (left_total / total) * imp_left + (right_total / total) * imp_right
            if weighted_imp < best_impurity - 1e-12:
                best_impurity = weighted_imp
                best_bias = float(0.5 * (proj_sorted[i] + proj_sorted[i + 1]))

        improvement = impurity_before - best_impurity
        return best_bias, improvement

    def _gini(self, count_pos: int, count_neg: int, w1: float, w0: float) -> float:
        pos = count_pos * w1
        neg = count_neg * w0
        total = pos + neg
        if total == 0:
            return 0.0
        p_pos = pos / total
        p_neg = neg / total
        return 1.0 - (p_pos**2 + p_neg**2)

    def _entropy(self, count_pos: int, count_neg: int, w1: float, w0: float) -> float:
        pos = count_pos * w1
        neg = count_neg * w0
        total = pos + neg
        if total == 0:
            return 0.0
        p_pos = pos / total
        p_neg = neg / total
        eps = 1e-12
        return -p_pos * math.log(p_pos + eps, 2) - p_neg * math.log(p_neg + eps, 2)

    def _misclass(self, count_pos: int, count_neg: int, w1: float, w0: float) -> float:
        pos = count_pos * w1
        neg = count_neg * w0
        total = pos + neg
        if total == 0:
            return 0.0
        p_pos = pos / total
        p_neg = neg / total
        return 1.0 - max(p_pos, p_neg)

    def _criterion_impurity(self, count_pos: int, count_neg: int, w1: float, w0: float, criterion: str) -> float:
        if criterion == "gini":
            return self._gini(count_pos, count_neg, w1, w0)
        if criterion == "gini_unweighted":
            return self._gini(count_pos, count_neg, 1.0, 1.0)
        if criterion == "entropy":
            return self._entropy(count_pos, count_neg, w1, w0)
        if criterion == "entropy_unweighted":
            return self._entropy(count_pos, count_neg, 1.0, 1.0)
        if criterion == "misclass":
            return self._misclass(count_pos, count_neg, w1, w0)
        if criterion == "misclass_unweighted":
            return self._misclass(count_pos, count_neg, 1.0, 1.0)
        if criterion in {"midpoint", "max_margin"}:
            return self._gini(count_pos, count_neg, w1, w0)
        return self._gini(count_pos, count_neg, w1, w0)

    def _is_redundant(self, n_vec: np.ndarray, bias: float) -> bool:
        if not self.hyperplanes:
            return False
        n1 = n_vec / (np.linalg.norm(n_vec) + 1e-12)
        projections = [np.dot(n1, pt) for pt in self.points]
        proj_range = 0.0
        if projections:
            proj_range = max(projections) - min(projections)
        for hp in self.hyperplanes:
            n2 = hp.normal / (np.linalg.norm(hp.normal) + 1e-12)
            cos_val = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
            if abs(cos_val) >= self.redundant_cos_thresh:
                # If we have points, require bias separation relative to projection span.
                if proj_range > 0.0:
                    if abs(bias - hp.bias) <= self.redundant_bias_frac * proj_range:
                        return True
                else:
                    # Fallback absolute check when no range (degenerate case)
                    if abs(bias - hp.bias) <= self.redundant_bias_frac:
                        return True
        return False

    def _build_cells_from_hyperplanes(self, hps: List[Hyperplane]) -> Tuple[Dict[int, CellStats], List[int]]:
        new_cells: Dict[int, CellStats] = {}
        patterns: List[int] = []
        for idx, (x, lbl) in enumerate(zip(self.points, self.labels)):
            bits = 0
            for i, hp in enumerate(hps):
                val = 1 if np.dot(hp.normal, x) >= hp.bias else 0
                bits |= val << i
            patterns.append(bits)
            cell = new_cells.setdefault(bits, CellStats())
            if lbl == 1:
                cell.support_1.append(idx)
            else:
                cell.support_0.append(idx)
        for pattern, cell in new_cells.items():
            for i, hp in enumerate(hps):
                bit = (pattern >> i) & 1
                if bit == 1 and hp.seed_1 is not None:
                    cell.seed_1.append(hp.seed_1)
                elif bit == 0 and hp.seed_0 is not None:
                    cell.seed_0.append(hp.seed_0)
        for cell in new_cells.values():
            cell.conflict_count = 0
        return new_cells, patterns

    def _accuracy_with_hyperplanes(self, hps: List[Hyperplane]) -> Tuple[float, Dict[int, CellStats], List[int]]:
        cells, patterns = self._build_cells_from_hyperplanes(hps)
        correct = 0
        for idx, pat in enumerate(patterns):
            cell = cells[pat]
            if cell.N == 0:
                seed_1 = len(cell.seed_1)
                seed_0 = len(cell.seed_0)
                if seed_1 + seed_0 == 0:
                    pred = 0
                else:
                    pred = 1 if seed_1 >= seed_0 else 0
            else:
                pred = cell.majority
            if pred == self.labels[idx]:
                correct += 1
        acc = correct / len(self.labels) if self.labels else 0.0
        return acc, cells, patterns

    def _rebuild_cells(self, reset_conflicts: bool = True) -> None:
        _, new_cells, patterns = self._accuracy_with_hyperplanes(self.hyperplanes)
        self.cells = new_cells

        # Track per-hyperplane side coverage.
        if self.hyperplanes:
            counts_0 = [0] * len(self.hyperplanes)
            counts_1 = [0] * len(self.hyperplanes)
            for pat in patterns:
                for i in range(len(self.hyperplanes)):
                    bit = (pat >> i) & 1
                    if bit:
                        counts_1[i] += 1
                    else:
                        counts_0[i] += 1
            for i, hp in enumerate(self.hyperplanes):
                if counts_0[i] > 0 and counts_1[i] > 0:
                    hp.last_two_sided_step = self.step


    def summary(self) -> str:
        lines = [
            f"Cells: {len(self.cells)} | Hyperplanes: {len(self.hyperplanes)}",
        ]
        for pat, cell in sorted(self.cells.items(), key=lambda kv: kv[0]):
            lines.append(
                f"pattern={pat:0{len(self.hyperplanes)}b} "
                f"N={cell.N} class1={cell.N_1} class0={cell.N_0} Δ={cell.delta:.3f} conflicts={cell.conflict_count}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def _normalize_label(self, y: int | float | str) -> int:
        if isinstance(y, str):
            y_lower = y.lower()
            if y_lower in {"a", "pos", "positive", "1"}:
                return 1
            if y_lower in {"b", "neg", "negative", "0"}:
                return 0
            raise ValueError(f"Unsupported label string '{y}'.")
        if y in (1, True):
            return 1
        if y in (0, False):
            return 0
        if y == -1:
            return 0
        raise ValueError(f"Unsupported label value '{y}'.")
