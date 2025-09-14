"""Real credit data: fetch, cache, clean, and split.

Source: the UCI *Default of Credit Card Clients* dataset (Taiwan, 2005) served
through OpenML -- 30,000 real borrowers, 23 features, a binary default flag with
a realistic ~22% positive rate. The raw frame is cached locally so only the
first run touches the network.
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from config import CANONICAL_COLUMNS, Config

LOGGER = logging.getLogger("credit.data")


def _canonical(X: pd.DataFrame) -> pd.DataFrame:
    """Rename generic x1..x23 columns to their UCI names (by fixed position)."""
    if list(X.columns) == [f"x{i}" for i in range(1, len(X.columns) + 1)] \
            and len(X.columns) == len(CANONICAL_COLUMNS):
        X = X.copy()
        X.columns = CANONICAL_COLUMNS
    return X


def load_credit(cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (features, binary target). Cached after the first download.

    Raises
    ------
    RuntimeError
        If the dataset cannot be fetched and no cache exists.
    """
    if os.path.exists(cfg.cache_path):
        with open(cfg.cache_path, "rb") as fh:
            X, y = pickle.load(fh)
        LOGGER.info("Loaded cached credit data: %s", X.shape)
        return _canonical(X), y

    try:
        bunch = fetch_openml(data_id=cfg.openml_id, as_frame=True, parser="auto")
    except Exception as exc:  # noqa: BLE001 - surface any network/OpenML error
        raise RuntimeError(
            f"Could not fetch OpenML dataset {cfg.openml_id} and no cache at "
            f"'{cfg.cache_path}'. Check the network. Original error: {exc}"
        ) from exc

    X = bunch.data.apply(pd.to_numeric, errors="coerce")
    # Target arrives as a categorical/string '0'/'1' -> clean integer {0, 1}.
    y = pd.to_numeric(bunch.target, errors="coerce").astype(int)
    y.name = "default"

    keep = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
    X = _canonical(X)

    with open(cfg.cache_path, "wb") as fh:
        pickle.dump((X, y), fh)
    LOGGER.info("Fetched + cached credit data: %s (default rate %.1f%%)",
                X.shape, 100.0 * y.mean())
    return X, y


def drop_protected_attributes(X: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Remove ECOA-protected demographic columns (fair-lending compliance)."""
    if not cfg.drop_protected:
        return X
    drop = [c for c in cfg.protected_cols if c in X.columns]
    return X.drop(columns=drop)


def split(X: pd.DataFrame, y: pd.Series, cfg: Config):
    """Stratified train/test split (credit data is cross-sectional, not a series)."""
    return train_test_split(X, y, test_size=cfg.test_size,
                            random_state=cfg.seed, stratify=y)
