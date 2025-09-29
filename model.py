"""Model contenders: a WOE+logistic scorecard and a gradient-boosting challenger.

Both are wrapped in ``CalibratedClassifierCV`` (isotonic) so the probabilities
they emit are genuine PDs -- essential, because a PD feeds expected loss
(``EL = PD·LGD·EAD``), not just an accept/decline gate. WOE lives *inside* the
pipeline, so calibration's internal CV refits it per fold with no leakage.
"""
from __future__ import annotations

from typing import Dict

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import Config
from woe import WOEBinner


def scorecard_pipeline(cfg: Config) -> Pipeline:
    """WOE binning -> logistic regression (the interpretable champion)."""
    return Pipeline([
        ("woe", WOEBinner(max_bins=cfg.woe_bins, max_cat=cfg.woe_max_cat,
                          smoothing=cfg.woe_smoothing)),
        ("lr", LogisticRegression(C=cfg.logreg_C, max_iter=cfg.max_iter)),
    ])


def build_models(cfg: Config) -> Dict[str, object]:
    """The two unfitted contenders."""
    gbm = HistGradientBoostingClassifier(
        max_iter=cfg.gb_max_iter, learning_rate=cfg.gb_learning_rate,
        max_depth=cfg.gb_max_depth, l2_regularization=cfg.gb_l2,
        random_state=cfg.seed)
    return {"Scorecard (WOE+LR)": scorecard_pipeline(cfg), "Gradient Boosting": gbm}


def calibrate(model, cfg: Config) -> CalibratedClassifierCV:
    """Wrap an estimator in post-hoc isotonic/Platt calibration (CV-fit)."""
    return CalibratedClassifierCV(model, method=cfg.calibration, cv=cfg.cv_folds)
