"""Headless scorecard + economics plots."""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from config import Config

LOGGER = logging.getLogger("credit.plot")


def plot_performance(scores: Dict[str, np.ndarray], y_true: np.ndarray,
                     uncal: Tuple[str, np.ndarray], cfg: Config, path: str) -> None:
    """ROC, PR, and a calibration panel showing the isotonic fix on the champion."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=cfg.dpi)
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        axes[0].plot(fpr, tpr, lw=1.6, label=name)
        prec, rec, _ = precision_recall_curve(y_true, s)
        axes[1].plot(rec, prec, lw=1.6, label=name)
        fp, mp = calibration_curve(y_true, s, n_bins=10, strategy="quantile")
        axes[2].plot(mp, fp, "o-", lw=1.4, ms=4, label=f"{name} (calibrated)")

    # the champion BEFORE calibration, to show the correction
    un_name, un_scores = uncal
    fp, mp = calibration_curve(y_true, un_scores, n_bins=10, strategy="quantile")
    axes[2].plot(mp, fp, "x--", lw=1.2, color="grey", label=f"{un_name} (uncalibrated)")

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    axes[0].set(title="ROC", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].axhline(float(np.mean(y_true)), color="k", ls="--", lw=0.8, alpha=0.6)
    axes[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    axes[2].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    axes[2].set(title="Calibration (isotonic fix)", xlabel="Mean predicted PD",
                ylabel="Observed default rate")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Credit PD Models - Discrimination & Calibration", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved performance plot -> %s", path)


def plot_scorecard(iv_df: pd.DataFrame, points: np.ndarray, y_true: np.ndarray,
                   gains: pd.DataFrame, cfg: Config, path: str) -> None:
    """Information Value, the score distribution by outcome, and the gains/lift curve."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=cfg.dpi)

    top = iv_df.head(12).iloc[::-1]
    axes[0].barh(top["feature"], top["iv"], color="#2c7fb8")
    axes[0].set(title="Information Value (feature strength)", xlabel="IV")
    axes[0].tick_params(axis="y", labelsize=8)

    axes[1].hist(points[y_true == 0], bins=40, alpha=0.6, density=True, label="Repaid")
    axes[1].hist(points[y_true == 1], bins=40, alpha=0.6, density=True, label="Defaulted")
    axes[1].set(title="Scorecard score by outcome", xlabel="Score (points)", ylabel="Density")
    axes[1].legend(frameon=False)

    axes[2].plot(gains["cum_pop"] * 100, gains["cum_capture"] * 100, "o-",
                 color="#2ca02c", lw=1.6, label="Model")
    axes[2].plot([0, 100], [0, 100], "k--", lw=0.8, alpha=0.6, label="Random")
    axes[2].set(title="Gains chart (bad capture)",
                xlabel="% population (worst PD first)", ylabel="% defaults captured")
    axes[2].legend(frameon=False); axes[2].grid(alpha=0.25)

    for ax in axes[:2]:
        ax.grid(alpha=0.25, axis="x")
    fig.suptitle("Scorecard Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved scorecard plot -> %s", path)


def plot_economics(curve: pd.DataFrame, optimal: Dict, cfg: Config, path: str) -> None:
    """Profit vs approval cut-off, and the approval-rate / bad-rate trade-off."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=cfg.dpi)

    ax = axes[0]
    ax.plot(curve["cutoff"], curve["profit"], color="#2ca02c", lw=1.8)
    ax.axvline(optimal["cutoff"], color="#d62728", ls="--", lw=1.2,
               label=f"Optimal PD cut-off {optimal['cutoff']:.2f}")
    ax.scatter([optimal["cutoff"]], [optimal["profit"]], color="#d62728", zorder=5)
    ax.set(title="Expected profit vs approval cut-off",
           xlabel="Approve if PD <= cut-off", ylabel="Realised profit ($, EAD units)")
    ax.legend(frameon=False); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(curve["cutoff"], curve["approval_rate"] * 100, color="#1f77b4", lw=1.6,
            label="Approval rate")
    ax.set_xlabel("Approve if PD <= cut-off")
    ax.set_ylabel("Approval rate (%)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(curve["cutoff"], curve["bad_rate_approved"] * 100, color="#d62728",
             lw=1.6, ls="--", label="Bad rate of approved book")
    ax2.set_ylabel("Bad rate of approved (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title("Approval rate vs quality of the approved book")
    ax.grid(alpha=0.25)

    fig.suptitle("Economic Decisioning", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved economics plot -> %s", path)
