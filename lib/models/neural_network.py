"""Multi-layer perceptron (neural network) classifier training."""

from typing import Any

import pandas as pd
from sklearn.neural_network import MLPClassifier  # type: ignore[import-untyped]

from lib.models.common import combine_train_val_sorted, grid_search_refit


def train_neural_network(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target_column: str = "target",
    *,
    param_grid: dict[str, list[Any]] | None = None,
    scoring: str = "f1",
    verbose: bool = False,
    grid_n_jobs: int = 1,
    **kwargs: Any,
) -> MLPClassifier:
    """Fit an ``MLPClassifier``. See ``train_decision_tree`` in ``decision_tree`` for grid search.

    Sensible defaults: ``max_iter=500``, ``early_stopping=True`` (internal holdout while fitting).
    Scale or z-score features before training; ``MLPClassifier`` is sensitive to feature scale.
    """

    def build(params: dict[str, Any], fit_data: pd.DataFrame) -> MLPClassifier:
        _ = fit_data
        kw = {
            "max_iter": 500,
            "early_stopping": True,
            "random_state": 42,
            **kwargs,
            **params,
        }
        return MLPClassifier(**kw)

    if param_grid:
        return grid_search_refit(
            build,
            param_grid,
            train_df,
            validation_df,
            target_column,
            scoring,
            verbose,
            grid_n_jobs=grid_n_jobs,
        )
    combined = combine_train_val_sorted(train_df, validation_df)
    features = combined.drop(columns=[target_column])
    targets = combined[target_column]
    merged = {
        "max_iter": 500,
        "early_stopping": True,
        "random_state": 42,
        **kwargs,
    }
    clf = MLPClassifier(**merged)
    clf.fit(features, targets)
    return clf
