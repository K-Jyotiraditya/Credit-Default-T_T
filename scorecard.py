"""Turn a fitted WOE + logistic model into a points scorecard and reason codes.

Siddiqi scaling maps log-odds to points: ``Score = offset + factor·ln(odds_good)``
with ``factor = PDO/ln2`` and ``offset`` pinned so the base score corresponds to
the base odds. Each (feature, bin) then contributes

    points = -(WOE·coef + intercept/n)·factor + offset/n

and the points sum to the applicant's score. Reason codes fall straight out: the
features costing the most points versus their best attainable bin.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from config import Config
from woe import WOEBinner


def build_scorecard(binner: WOEBinner, coef: np.ndarray, intercept: float,
                    cfg: Config) -> pd.DataFrame:
    """Scorecard table: one row per (feature, bin) with its WOE and point value."""
    factor = cfg.pdo / np.log(2.0)
    offset = cfg.base_score - factor * np.log(cfg.base_odds)
    n = len(binner.columns_)

    rows = []
    for j, col in enumerate(binner.columns_):
        woe = binner.maps_[col]["woe"]
        labels = binner.bin_labels(col)
        for b, (lab, w) in enumerate(zip(labels, woe)):
            points = -(w * coef[j] + intercept / n) * factor + offset / n
            rows.append({"feature": col, "bin": lab, "woe": float(w),
                         "points": float(points)})
    return pd.DataFrame(rows)


def score_applicants(binner: WOEBinner, scorecard: pd.DataFrame,
                     X: pd.DataFrame) -> np.ndarray:
    """Total scorecard points per applicant (higher score = lower risk)."""
    X = pd.DataFrame(X)
    pts = np.zeros(len(X))
    for col in binner.columns_:
        m = binner.maps_[col]
        x = X[col].to_numpy()
        sub = scorecard[scorecard["feature"] == col].reset_index(drop=True)
        if m["kind"] == "cat":
            lookup = {str(c): sub.loc[i, "points"] for i, c in enumerate(m["cats"])}
            pts += np.array([lookup.get(str(v), 0.0) for v in x])
        else:
            idx = np.clip(np.digitize(x, m["edges"][1:-1]), 0, len(sub) - 1)
            pts += sub["points"].to_numpy()[idx]
    return pts


def score_to_pd(score: np.ndarray, cfg: Config) -> np.ndarray:
    """Invert the scaling: score -> implied probability of default."""
    odds_good = cfg.base_odds * 2.0 ** ((np.asarray(score) - cfg.base_score) / cfg.pdo)
    return 1.0 / (1.0 + odds_good)


def reason_codes(binner: WOEBinner, scorecard: pd.DataFrame, X: pd.DataFrame,
                 top_k: int = 3) -> List[List[str]]:
    """Per-applicant adverse-action reasons: features losing the most points."""
    X = pd.DataFrame(X)
    best = {col: scorecard.loc[scorecard["feature"] == col, "points"].max()
            for col in binner.columns_}
    reasons: List[List[str]] = []
    for _, row in X.iterrows():
        gaps: Dict[str, float] = {}
        for col in binner.columns_:
            m = binner.maps_[col]
            v = row[col]
            sub = scorecard[scorecard["feature"] == col].reset_index(drop=True)
            if m["kind"] == "cat":
                k = list(m["cats"]).index(v) if v in m["cats"] else 0
            else:
                k = int(np.clip(np.digitize([v], m["edges"][1:-1])[0], 0, len(sub) - 1))
            gaps[col] = best[col] - sub.loc[k, "points"]      # points forgone
        top = sorted(gaps, key=gaps.get, reverse=True)[:top_k]
        reasons.append(top)
    return reasons
