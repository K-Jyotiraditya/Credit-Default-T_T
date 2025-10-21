"""From PD to a lending decision: expected loss, profit, and the optimal cut-off.

A PD is only worth money once it drives an approve/decline rule. We approve when
the predicted default probability is below a cut-off, then score the *realised*
economics on the test book: a good account earns ``revenue_rate * EAD`` and a
default costs ``LGD * EAD`` (EAD proxied by the credit limit). Sweeping the
cut-off traces the profit curve and locates the profit-maximising policy --
which is generally *not* the accuracy-maximising threshold.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config import Config


def expected_loss(pd_hat: np.ndarray, ead: np.ndarray, cfg: Config) -> np.ndarray:
    """Per-account expected loss EL = PD * LGD * EAD."""
    return np.asarray(pd_hat) * cfg.lgd * np.asarray(ead)


def profit_curve(y_true: np.ndarray, pd_hat: np.ndarray, ead: np.ndarray,
                 cfg: Config) -> pd.DataFrame:
    """Realised approve-book profit as the PD approval cut-off is swept."""
    y, pd_hat, ead = np.asarray(y_true), np.asarray(pd_hat), np.asarray(ead, float)
    rows = []
    for cutoff in np.linspace(0.02, 0.98, 49):
        approve = pd_hat <= cutoff
        if not approve.any():
            rows.append({"cutoff": cutoff, "approval_rate": 0.0,
                         "bad_rate_approved": 0.0, "profit": 0.0})
            continue
        good = approve & (y == 0)
        bad = approve & (y == 1)
        profit = cfg.revenue_rate * ead[good].sum() - cfg.lgd * ead[bad].sum()
        rows.append({
            "cutoff": float(cutoff),
            "approval_rate": float(approve.mean()),
            "bad_rate_approved": float(y[approve].mean()),
            "profit": float(profit),
        })
    return pd.DataFrame(rows)


def optimal_cutoff(curve: pd.DataFrame) -> Dict:
    """The profit-maximising row of the profit curve."""
    best = curve.loc[curve["profit"].idxmax()]
    return {"cutoff": float(best["cutoff"]), "profit": float(best["profit"]),
            "approval_rate": float(best["approval_rate"]),
            "bad_rate_approved": float(best["bad_rate_approved"])}
