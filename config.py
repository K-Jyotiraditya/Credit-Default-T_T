"""Run configuration for the credit-default scorecard development."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """Hyper-parameters and I/O knobs for one scorecard run."""

    # --- Data (real: UCI 'default of credit card clients', via OpenML) ------
    openml_id: int = 42477
    cache_path: str = "credit_data.pkl"
    test_size: float = 0.20
    seed: int = 42
    ead_col: str = "LIMIT_BAL"          # exposure-at-default proxy

    # Fair lending: SEX, MARRIAGE, AGE are protected under ECOA (EDUCATION is a
    # common proxy); we exclude them so the model -- and its reason codes -- rest
    # on behavioural/financial features only.
    drop_protected: bool = True
    protected_cols: tuple = ("SEX", "MARRIAGE", "AGE", "EDUCATION")

    # --- WOE / Information-Value binning -----------------------------------
    woe_bins: int = 8                   # max quantile bins for continuous features
    woe_max_cat: int = 11               # <= this many uniques -> treat as categorical
    woe_smoothing: float = 0.5          # Laplace count to avoid 0/inf WOE
    iv_floor: float = 0.02              # drop features below this Information Value

    # --- Scorecard scaling (Siddiqi points) --------------------------------
    pdo: float = 20.0                   # points to double the (good:bad) odds
    base_score: float = 600.0
    base_odds: float = 50.0             # good:bad odds at the base score

    # --- Models / calibration / CV -----------------------------------------
    logreg_C: float = 1.0
    max_iter: int = 2000
    gb_max_iter: int = 300
    gb_learning_rate: float = 0.05
    gb_max_depth: int = 3
    gb_l2: float = 1.0
    cv_folds: int = 5
    calibration: str = "isotonic"       # post-hoc probability calibration

    # --- Decision threshold + economics ------------------------------------
    fn_cost: float = 5.0                # cost of approving a future default
    fp_cost: float = 1.0                # cost of rejecting a good client
    revenue_rate: float = 0.10          # revenue on EAD from a good account
    lgd: float = 0.75                   # loss given default (fraction of EAD)

    # --- Plotting ----------------------------------------------------------
    dpi: int = 150


# Canonical names for the UCI 'default of credit card clients' columns, in the
# dataset's fixed order, so the scorecard and reason codes are human-readable.
CANONICAL_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
