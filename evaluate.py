"""Credit-risk evaluation metrics and cost-based thresholding.

Discrimination is summarised with the metrics a credit desk actually quotes --
AUC/Gini and the KS statistic -- plus probability-quality (Brier) and a
cost-optimal cut-off, because a PD model is only useful once turned into an
approve/decline decision under asymmetric costs.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from config import Config

LOGGER = logging.getLogger("credit.eval")


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max gap between the good/bad score CDFs (0..1).

    Ties are grouped (the gap is only read at the end of each run of equal
    scores), so a non-discriminating model with constant scores scores 0.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    cdf_pos = np.cumsum(y_sorted) / n_pos
    cdf_neg = np.cumsum(1 - y_sorted) / n_neg
    last_of_tie = np.append(s_sorted[1:] != s_sorted[:-1], True)
    return float(np.max(np.abs(cdf_pos - cdf_neg)[last_of_tie]))


def gini(auc: float) -> float:
    """Gini coefficient = 2*AUC - 1 (the credit-industry rank-ordering score)."""
    return 2.0 * auc - 1.0


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, cfg: Config) -> float:
    """Pick the probability cut-off that minimises expected misclassification cost."""
    y = np.asarray(y_true)
    grid = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, np.inf
    for t in grid:
        pred = scores >= t
        fn = int(((y == 1) & (~pred)).sum())
        fp = int(((y == 0) & (pred)).sum())
        cost = cfg.fn_cost * fn + cfg.fp_cost * fp
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t


def evaluate_model(name: str, y_true: np.ndarray, scores: np.ndarray,
                   cfg: Config) -> Dict:
    """Full metric bundle for one model's probability predictions."""
    y = np.asarray(y_true)
    auc = roc_auc_score(y, scores)
    threshold = choose_threshold(y, scores, cfg)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    result = {
        "name": name,
        "auc": float(auc),
        "gini": gini(float(auc)),
        "pr_auc": float(average_precision_score(y, scores)),
        "ks": ks_statistic(y, scores),
        "brier": float(brier_score_loss(y, scores)),
        "threshold": threshold,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "cost": float(cfg.fn_cost * fn + cfg.fp_cost * fp),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    LOGGER.info("%-20s | AUC %.3f | Gini %.3f | KS %.3f | Brier %.3f | "
                "recall %.3f @ t=%.2f", name, result["auc"], result["gini"],
                result["ks"], result["brier"], result["recall"], threshold)
    return result


def cross_validate_model(model, X, y, cfg: Config) -> Dict:
    """Stratified K-fold AUC / Gini / KS (mean +/- std) -- a robust read, not one split."""
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    aucs, kss = [], []
    Xv = X.reset_index(drop=True)
    yv = np.asarray(y)
    for tr, va in skf.split(Xv, yv):
        est = clone(model)
        est.fit(Xv.iloc[tr], yv[tr])
        s = est.predict_proba(Xv.iloc[va])[:, 1]
        aucs.append(roc_auc_score(yv[va], s))
        kss.append(ks_statistic(yv[va], s))
    aucs, kss = np.array(aucs), np.array(kss)
    return {"auc_mean": float(aucs.mean()), "auc_std": float(aucs.std()),
            "gini_mean": float(2 * aucs.mean() - 1), "ks_mean": float(kss.mean()),
            "ks_std": float(kss.std()), "folds": cfg.cv_folds}


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two score distributions.

    <0.10 stable, 0.10-0.25 minor shift, >0.25 material shift. Bins are fixed on
    the *expected* (development) distribution.
    """
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def gains_table(y_true: np.ndarray, pd_hat: np.ndarray, n_bands: int = 10) -> pd.DataFrame:
    """Decile gains/lift: rank by PD, then bad-rate, cumulative capture, and lift."""
    df = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(pd_hat)})
    df = df.sort_values("pd", ascending=False).reset_index(drop=True)
    df["band"] = (np.arange(len(df)) * n_bands // len(df)) + 1
    base_rate = df["y"].mean()
    g = df.groupby("band").agg(n=("y", "size"), bads=("y", "sum")).reset_index()
    g["bad_rate"] = g["bads"] / g["n"]
    g["cum_capture"] = g["bads"].cumsum() / df["y"].sum()
    g["cum_pop"] = g["n"].cumsum() / len(df)
    g["lift"] = g["bad_rate"] / base_rate
    return g


def format_scoreboard(results: Dict[str, Dict]) -> str:
    """Side-by-side metric table for all models."""
    head = (f"{'Model':<22}{'AUC':>7}{'Gini':>7}{'PR-AUC':>8}{'KS':>7}"
            f"{'Brier':>8}{'Recall':>8}{'Prec':>7}{'Thr':>6}")
    lines = [head, "-" * len(head)]
    for r in results.values():
        lines.append(f"{r['name']:<22}{r['auc']:>7.3f}{r['gini']:>7.3f}"
                     f"{r['pr_auc']:>8.3f}{r['ks']:>7.3f}{r['brier']:>8.3f}"
                     f"{r['recall']:>8.3f}{r['precision']:>7.3f}{r['threshold']:>6.2f}")
    return "\n".join(lines)
