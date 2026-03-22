"""Gaussian Naive Bayes classifier training."""

from typing import Any

import pandas as pd
from sklearn.naive_bayes import GaussianNB  # type: ignore[import-untyped]

from lib.models.common import combine_train_val_sorted, grid_search_refit


def train_naive_bayes(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target_column: str = "target",
    *,
    param_grid: dict[str, list[Any]] | None = None,
    scoring: str = "f1",
    verbose: bool = False,
    grid_n_jobs: int = 1,
    **kwargs: Any,
) -> GaussianNB:
    """Fit a Gaussian Naive Bayes classifier. See ``train_decision_tree`` in ``decision_tree`` for grid search."""

    def build(params: dict[str, Any], fit_data: pd.DataFrame) -> GaussianNB:
        _ = fit_data
        return GaussianNB(**{**kwargs, **params})

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
    clf = GaussianNB(**kwargs)
    clf.fit(features, targets)
    return clf
