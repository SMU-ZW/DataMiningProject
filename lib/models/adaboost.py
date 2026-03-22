"""AdaBoost classifier training."""

from typing import Any

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier  # type: ignore[import-untyped]
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import-untyped]

from lib.models.common import combine_train_val_sorted, grid_search_refit


def train_adaboost(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target_column: str = "target",
    *,
    param_grid: dict[str, list[Any]] | None = None,
    scoring: str = "f1",
    verbose: bool = False,
    grid_n_jobs: int = 1,
    **kwargs: Any,
) -> AdaBoostClassifier:
    """Fit an AdaBoost classifier. Uses a shallow tree (max_depth=3) as default base estimator.

    See ``train_decision_tree`` in ``decision_tree`` for grid search vs single-fit behavior.
    """

    def build(params: dict[str, Any], fit_data: pd.DataFrame) -> AdaBoostClassifier:
        kw = {**kwargs, **params}
        if "estimator" not in kw:
            kw["estimator"] = DecisionTreeClassifier(
                max_depth=3, random_state=42, class_weight="balanced"
            )
        _ = fit_data
        return AdaBoostClassifier(random_state=42, **kw)

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
    if "estimator" not in kwargs:
        kwargs["estimator"] = DecisionTreeClassifier(
            max_depth=3, random_state=42, class_weight="balanced"
        )
    clf = AdaBoostClassifier(random_state=42, **kwargs)
    clf.fit(features, targets)
    return clf
