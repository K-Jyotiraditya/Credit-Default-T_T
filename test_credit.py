"""Offline unit tests for the scorecard pipeline (tiny synthetic frames)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import data as data_mod
from config import Config
from economics import expected_loss, optimal_cutoff, profit_curve
from evaluate import (choose_threshold, cross_validate_model, gains_table, gini,
                      ks_statistic, psi)
from model import build_models, calibrate, scorecard_pipeline
from scorecard import (build_scorecard, reason_codes, score_applicants,
                       score_to_pd)
from woe import WOEBinner


def _toy(n: int = 900, seed: int = 0):
    rng = np.random.default_rng(seed)
    x1, x3 = rng.normal(size=n), rng.normal(size=n)          # signal, noise
    x2 = rng.integers(0, 3, n)                               # coded/categorical
    logit = 1.5 * x1 + 0.8 * (x2 == 2) - 0.6
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}), pd.Series(y, name="default")


# --------------------------------------------------------------------------- #
# WOE / IV
# --------------------------------------------------------------------------- #
def test_woe_iv_ranks_signal_above_noise():
    X, y = _toy()
    b = WOEBinner().fit(X, y)
    assert b.iv_["x1"] > b.iv_["x3"]                 # signal beats noise
    assert b.transform(X).shape == (len(X), 3)


def test_woe_high_iv_on_separable_feature():
    x = np.linspace(-3, 3, 600)
    X = pd.DataFrame({"f": x})
    y = pd.Series((x > 0).astype(int))               # perfectly separable
    b = WOEBinner().fit(X, y)
    assert b.iv_["f"] > 0.5
    assert np.all(np.isfinite(b.transform(X)))       # smoothing prevents inf


def test_woe_is_sklearn_pipeline_compatible():
    X, y = _toy()
    pipe = scorecard_pipeline(Config()).fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert np.all((p >= 0) & (p <= 1))


# --------------------------------------------------------------------------- #
# Scorecard scaling + reason codes
# --------------------------------------------------------------------------- #
def test_scorecard_pdo_doubles_odds():
    cfg = Config()
    p0 = score_to_pd(np.array([cfg.base_score]), cfg)[0]
    p1 = score_to_pd(np.array([cfg.base_score + cfg.pdo]), cfg)[0]
    assert p0 == pytest.approx(1 / (1 + cfg.base_odds))
    odds0, odds1 = (1 - p0) / p0, (1 - p1) / p1
    assert odds1 == pytest.approx(2 * odds0, rel=1e-6)       # +PDO -> 2x odds


def test_scorecard_points_separate_outcomes():
    X, y = _toy()
    sc = scorecard_pipeline(Config()).fit(X, y)
    b, lr = sc.named_steps["woe"], sc.named_steps["lr"]
    table = build_scorecard(b, lr.coef_[0], lr.intercept_[0], Config())
    pts = score_applicants(b, table, X)
    assert len(pts) == len(X)
    assert pts[y.to_numpy() == 0].mean() > pts[y.to_numpy() == 1].mean()   # good scores higher


def test_reason_codes_return_k_features():
    X, y = _toy()
    sc = scorecard_pipeline(Config()).fit(X, y)
    b, lr = sc.named_steps["woe"], sc.named_steps["lr"]
    table = build_scorecard(b, lr.coef_[0], lr.intercept_[0], Config())
    codes = reason_codes(b, table, X.iloc[:5], top_k=2)
    assert len(codes) == 5 and all(len(c) == 2 for c in codes)


# --------------------------------------------------------------------------- #
# Evaluation: KS, PSI, gains, CV, calibration
# --------------------------------------------------------------------------- #
def test_ks_tie_safe_and_gini():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert ks_statistic(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)
    assert gini(0.5) == pytest.approx(0.0)


def test_psi_zero_identical_positive_shifted():
    base = np.random.default_rng(0).normal(0, 1, 5000)
    assert psi(base, base) == pytest.approx(0.0, abs=1e-9)
    assert psi(base, base + 1.5) > 0.25                     # large shift flagged


def test_gains_table_captures_all_bads():
    X, y = _toy()
    pd_hat = np.random.default_rng(1).uniform(size=len(y))
    g = gains_table(y.to_numpy(), pd_hat)
    assert g["n"].sum() == len(y)
    assert g["cum_capture"].iloc[-1] == pytest.approx(1.0)


def test_cross_validate_reports_folds():
    X, y = _toy()
    cv = cross_validate_model(scorecard_pipeline(Config()), X, y, Config())
    assert cv["folds"] == Config().cv_folds
    assert 0.5 < cv["auc_mean"] <= 1.0


def test_calibration_outputs_valid_proba():
    X, y = _toy()
    cal = calibrate(scorecard_pipeline(Config()), Config()).fit(X, y)
    p = cal.predict_proba(X)[:, 1]
    assert np.all((p >= 0) & (p <= 1))


def test_choose_threshold_in_range():
    X, y = _toy()
    t = choose_threshold(y.to_numpy(), np.random.default_rng(0).uniform(size=len(y)), Config())
    assert 0.0 < t < 1.0


# --------------------------------------------------------------------------- #
# Economics
# --------------------------------------------------------------------------- #
def test_expected_loss_elementwise():
    cfg = Config()
    el = expected_loss(np.array([0.1, 0.5]), np.array([1000.0, 2000.0]), cfg)
    assert el[0] == pytest.approx(0.1 * cfg.lgd * 1000.0)


def test_profit_curve_optimal_is_max():
    X, y = _toy()
    pd_hat = np.random.default_rng(2).uniform(0, 0.4, size=len(y))
    ead = np.full(len(y), 1000.0)
    curve = profit_curve(y.to_numpy(), pd_hat, ead, Config())
    opt = optimal_cutoff(curve)
    assert opt["profit"] == pytest.approx(curve["profit"].max())
    assert 0.0 < opt["cutoff"] < 1.0


# --------------------------------------------------------------------------- #
# Data loader cache
# --------------------------------------------------------------------------- #
def test_load_credit_uses_cache(tmp_path, monkeypatch):
    X, y = _toy(60)

    class _Bunch:
        pass

    bunch = _Bunch()
    bunch.data, bunch.target = X, y.astype(str)
    monkeypatch.setattr(data_mod, "fetch_openml", lambda **k: bunch)
    cfg = Config(cache_path=str(tmp_path / "credit.pkl"))
    X1, _ = data_mod.load_credit(cfg)
    assert os.path.exists(cfg.cache_path)

    def _boom(**k):
        raise RuntimeError("network down")

    monkeypatch.setattr(data_mod, "fetch_openml", _boom)
    X2, _ = data_mod.load_credit(cfg)
    assert X2.shape == X1.shape
