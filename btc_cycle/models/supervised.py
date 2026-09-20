"""Supervised phase classification, validated honestly.

A classifier on these features gets ~90% daily accuracy under random k-fold CV.
That number is fake twice over: adjacent days are near-identical (so the test
set is in the training set), and random folds mean the model has seen the cycle
it is being graded on.

:func:`leave_one_cycle_out` is the version worth reading. It trains on complete
cycles and predicts a held-out one, which is the real task - and it reports the
effective sample size, which is 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, labels


DEFAULT_FEATURES = [
    "log_mayer",
    "log_ma200w_mult",
    "ret_90",
    "ret_365",
    "realized_vol_30",
    "drawdown",
]


def _design(frame: pd.DataFrame, feature_names: list) -> tuple:
    usable = [n for n in feature_names if n in frame.columns]
    y = labels.label_phases(frame["price"])
    data = pd.concat([frame[usable], y.rename("phase")], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data["cycle"] = labels.cycle_id(data.index)
    return data, usable


def leave_one_cycle_out(frame: pd.DataFrame, feature_names: list | None = None,
                        model_name: str = "logistic") -> dict:
    """Train on all complete cycles but one, predict the held-out cycle."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier

    feature_names = feature_names or DEFAULT_FEATURES
    data, usable = _design(frame, feature_names)
    complete = {float(c.index) for c in config.COMPLETED_CYCLES}
    data = data[data["cycle"].isin(complete)]
    if data.empty:
        return {"error": "no labelled rows"}

    folds = []
    for held_out in sorted(data["cycle"].unique()):
        train = data[data["cycle"] != held_out]
        test = data[data["cycle"] == held_out]
        if train.empty or test.empty or train["phase"].nunique() < 2:
            continue

        mean = train[usable].mean()
        std = train[usable].std().replace(0.0, 1.0)
        if model_name == "logistic":
            model = LogisticRegression(max_iter=2000, C=1.0)
        else:
            model = GradientBoostingClassifier(random_state=0, n_estimators=150)
        model.fit((train[usable] - mean) / std, train["phase"])
        predicted = model.predict((test[usable] - mean) / std)

        accuracy = float((predicted == test["phase"].to_numpy()).mean())
        base_rate = float(test["phase"].value_counts(normalize=True).max())
        folds.append(
            {
                "held_out_cycle": int(held_out),
                "n_test_days": int(len(test)),
                "accuracy": accuracy,
                "majority_class_rate": base_rate,
                "lift_over_majority": accuracy - base_rate,
            }
        )

    if not folds:
        return {"error": "no usable folds"}

    table = pd.DataFrame(folds)
    return {
        "folds": folds,
        "mean_accuracy": float(table["accuracy"].mean()),
        "mean_lift": float(table["lift_over_majority"].mean()),
        "effective_sample_size": len(folds),
        "warning": (
            f"{len(folds)} independent cycles. A standard error on 3 points is "
            "not a standard error. Read the per-fold spread, not the mean."
        ),
    }


def random_kfold_for_contrast(frame: pd.DataFrame,
                              feature_names: list | None = None,
                              n_splits: int = 5) -> dict:
    """The inflated number, computed on purpose so the gap is visible."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    feature_names = feature_names or DEFAULT_FEATURES
    data, usable = _design(frame, feature_names)
    if data.empty:
        return {"error": "no labelled rows"}
    mean = data[usable].mean()
    std = data[usable].std().replace(0.0, 1.0)
    scores = cross_val_score(
        LogisticRegression(max_iter=2000),
        (data[usable] - mean) / std,
        data["phase"],
        cv=n_splits,
    )
    return {
        "random_kfold_accuracy": float(scores.mean()),
        "note": (
            "Inflated: adjacent days are near-duplicates and folds span cycles, "
            "so the model is graded on data it effectively trained on. Compare "
            "against leave_one_cycle_out, not against this."
        ),
    }


def fit_current(frame: pd.DataFrame, feature_names: list | None = None) -> dict:
    """Train on every complete cycle, then read the running cycle's latest day."""
    from sklearn.linear_model import LogisticRegression

    feature_names = feature_names or DEFAULT_FEATURES
    data, usable = _design(frame, feature_names)
    complete = {float(c.index) for c in config.COMPLETED_CYCLES}
    train = data[data["cycle"].isin(complete)]
    if train.empty:
        return {"error": "no labelled training rows"}

    mean = train[usable].mean()
    std = train[usable].std().replace(0.0, 1.0)
    model = LogisticRegression(max_iter=2000)
    model.fit((train[usable] - mean) / std, train["phase"])

    live = frame[usable].replace([np.inf, -np.inf], np.nan).dropna()
    if live.empty:
        return {"error": "no usable live rows"}
    latest = live.iloc[[-1]]
    probabilities = model.predict_proba((latest - mean) / std)[0]

    return {
        "as_of": str(latest.index[-1].date()),
        "probabilities": {
            phase: float(p) for phase, p in zip(model.classes_, probabilities)
        },
        "features_used": usable,
    }
