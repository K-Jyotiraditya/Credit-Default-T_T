"""Driver: real data -> WOE scorecard + GBM -> calibrate -> CV -> economics -> plots.

    python main.py            # downloads + caches on first call, then runs
    python -m pytest -q       # offline unit tests

Champion = the interpretable WOE+logistic scorecard (what a bank deploys for
explainability/regulatory reasons); GBM is the challenger benchmark.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss

from config import Config
from data import drop_protected_attributes, load_credit, split
from economics import optimal_cutoff, profit_curve
from evaluate import (cross_validate_model, evaluate_model, format_scoreboard,
                      gains_table, psi)
from model import build_models, calibrate, scorecard_pipeline
from plotting import plot_economics, plot_performance, plot_scorecard
from scorecard import build_scorecard, reason_codes, score_applicants

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
LOGGER = logging.getLogger("credit")
HERE = Path(__file__).resolve().parent
CHAMPION = "Scorecard (WOE+LR)"


def run(cfg: Config) -> dict:
    X, y = load_credit(cfg)
    X = drop_protected_attributes(X, cfg)
    LOGGER.info("Dataset: %d rows, %d features (protected dropped), default rate %.1f%%",
                len(X), X.shape[1], 100.0 * y.mean())
    X_tr, X_te, y_tr, y_te = split(X, y, cfg)
    yte = y_te.to_numpy()

    # --- Cross-validated discrimination (robust, not one split) ---
    models = build_models(cfg)
    print("\nCross-validated discrimination (train, %d folds):" % cfg.cv_folds)
    for name, model in models.items():
        cv = cross_validate_model(model, X_tr, y_tr, cfg)
        print(f"  {name:<22} AUC {cv['auc_mean']:.3f} +/- {cv['auc_std']:.3f} | "
              f"Gini {cv['gini_mean']:.3f} | KS {cv['ks_mean']:.3f} +/- {cv['ks_std']:.3f}")

    # --- Fit, calibrate, and evaluate each model on the test set ---
    scores, results, brier_uncal = {}, {}, {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        uncal = model.predict_proba(X_te)[:, 1]
        cal = calibrate(build_models(cfg)[name], cfg).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        brier_uncal[name] = brier_score_loss(yte, uncal)
        scores[name] = cal
        results[name] = evaluate_model(name, yte, cal, cfg)
    print("\n" + format_scoreboard(results))
    print(f"\nCalibration (Brier, lower=better): "
          f"{CHAMPION} {brier_uncal[CHAMPION]:.4f} -> {results[CHAMPION]['brier']:.4f} after isotonic")

    # --- Build the points scorecard from the fitted WOE + logistic ---
    sc = scorecard_pipeline(cfg).fit(X_tr, y_tr)
    binner = sc.named_steps["woe"]
    lr = sc.named_steps["lr"]
    scorecard_df = build_scorecard(binner, lr.coef_[0], lr.intercept_[0], cfg)
    iv_df = binner.iv_table()
    pts_tr = score_applicants(binner, scorecard_df, X_tr)
    pts_te = score_applicants(binner, scorecard_df, X_te)

    print("\nTop features by Information Value:")
    for _, r in iv_df.head(6).iterrows():
        print(f"  {r['feature']:<12} IV {r['iv']:.3f}  ({r['strength']})")

    # --- Stability (PSI) of the score between development and test ---
    score_psi = psi(pts_tr, pts_te)
    LOGGER.info("Score PSI (train vs test): %.4f (%s)", score_psi,
                "stable" if score_psi < 0.1 else "shift")

    # --- Economic decisioning on the champion's calibrated PD ---
    ead = X_te[cfg.ead_col].to_numpy()
    curve = profit_curve(yte, scores[CHAMPION], ead, cfg)
    opt = optimal_cutoff(curve)
    LOGGER.info("Profit-max cut-off: approve if PD<=%.2f -> approve %.0f%%, "
                "approved bad-rate %.1f%%", opt["cutoff"], opt["approval_rate"] * 100,
                opt["bad_rate_approved"] * 100)

    gains = gains_table(yte, scores[CHAMPION])
    print(f"\nTop-decile lift: {gains.iloc[0]['lift']:.2f}x base rate | "
          f"top 30% of PD captures {gains.iloc[2]['cum_capture']*100:.0f}% of defaults")

    # --- Reason codes for a few declined applicants ---
    declined = np.where(scores[CHAMPION] > opt["cutoff"])[0][:3]
    codes = reason_codes(binner, scorecard_df, X_te.iloc[declined])
    print("\nSample adverse-action reason codes (declined applicants):")
    for i, c in zip(declined, codes):
        print(f"  PD {scores[CHAMPION][i]:.2f} -> {', '.join(c)}")

    # --- Plots ---
    plot_performance(scores, yte, (CHAMPION, sc.predict_proba(X_te)[:, 1]),
                     cfg, str(HERE / "pd_curves.png"))
    plot_scorecard(iv_df, pts_te, yte, gains, cfg, str(HERE / "scorecard_analysis.png"))
    plot_economics(curve, opt, cfg, str(HERE / "economics.png"))
    return {"results": results, "iv": iv_df, "optimal": opt, "psi": score_psi}


def main() -> int:
    try:
        run(Config())
    except RuntimeError as exc:
        LOGGER.error("Aborted: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
