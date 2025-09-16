"""Weight-of-Evidence binning and Information Value -- the scorecard front end.

WOE recodes every feature as the log-odds of *good* vs *bad* within a bin:

    WOE_b = ln( (good_b / total_good) / (bad_b / total_bad) )

so a single monotone, missing-safe, outlier-robust number replaces the raw value.
Information Value, `IV = Σ_b (g_b − b_b)·WOE_b`, ranks features by predictive
power. Implemented as a scikit-learn transformer so it is refit *inside* every
cross-validation fold -- WOE fit on test data would leak the target.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WOEBinner(BaseEstimator, TransformerMixin):
    """Supervised WOE binning: quantile bins for continuous, value bins for coded."""

    def __init__(self, max_bins: int = 8, max_cat: int = 11, smoothing: float = 0.5):
        self.max_bins = max_bins
        self.max_cat = max_cat
        self.smoothing = smoothing

    # -- helpers ------------------------------------------------------------
    def _woe_of_counts(self, good: np.ndarray, bad: np.ndarray,
                       tot_good: int, tot_bad: int):
        n = len(good)
        dist_good = (good + self.smoothing) / (tot_good + self.smoothing * n)
        dist_bad = (bad + self.smoothing) / (tot_bad + self.smoothing * n)
        woe = np.log(dist_good / dist_bad)
        iv = float(np.sum((dist_good - dist_bad) * woe))
        return woe, iv

    # -- sklearn API --------------------------------------------------------
    def fit(self, X: pd.DataFrame, y) -> "WOEBinner":
        X = pd.DataFrame(X)
        y = np.asarray(y).astype(int)
        self.columns_ = list(X.columns)
        tot_good, tot_bad = int((y == 0).sum()), int((y == 1).sum())
        self.maps_: Dict[str, dict] = {}
        self.iv_: Dict[str, float] = {}

        for col in self.columns_:
            x = X[col].to_numpy()
            if X[col].nunique() <= self.max_cat:
                cats = np.sort(X[col].unique())
                good = np.array([((x == c) & (y == 0)).sum() for c in cats])
                bad = np.array([((x == c) & (y == 1)).sum() for c in cats])
                woe, iv = self._woe_of_counts(good, bad, tot_good, tot_bad)
                self.maps_[col] = {"kind": "cat", "cats": cats, "woe": woe}
            else:
                qs = np.unique(np.quantile(x, np.linspace(0, 1, self.max_bins + 1)))
                edges = np.concatenate([[-np.inf], qs[1:-1], [np.inf]])
                idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
                nb = len(edges) - 1
                good = np.array([((idx == b) & (y == 0)).sum() for b in range(nb)])
                bad = np.array([((idx == b) & (y == 1)).sum() for b in range(nb)])
                woe, iv = self._woe_of_counts(good, bad, tot_good, tot_bad)
                self.maps_[col] = {"kind": "num", "edges": edges, "woe": woe}
            self.iv_[col] = iv
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        out = np.zeros((len(X), len(self.columns_)), dtype=float)
        for j, col in enumerate(self.columns_):
            m = self.maps_[col]
            x = X[col].to_numpy()
            if m["kind"] == "cat":
                lookup = {c: w for c, w in zip(m["cats"], m["woe"])}
                out[:, j] = np.array([lookup.get(v, 0.0) for v in x])
            else:
                idx = np.clip(np.digitize(x, m["edges"][1:-1]), 0, len(m["woe"]) - 1)
                out[:, j] = m["woe"][idx]
        return out

    # -- reporting ----------------------------------------------------------
    def iv_table(self) -> pd.DataFrame:
        """Information Value per feature, sorted, with the usual strength labels."""
        def strength(iv):
            return ("unpredictive" if iv < 0.02 else "weak" if iv < 0.1 else
                    "medium" if iv < 0.3 else "strong" if iv < 0.5 else "suspicious")
        rows = [{"feature": c, "iv": v, "strength": strength(v)}
                for c, v in self.iv_.items()]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def bin_labels(self, col: str) -> List[str]:
        """Readable bin labels for the scorecard table."""
        m = self.maps_[col]
        if m["kind"] == "cat":
            return [str(c) for c in m["cats"]]
        e = m["edges"]
        return [f"({e[b]:.0f}, {e[b + 1]:.0f}]" for b in range(len(e) - 1)]
