"""Random forest classifier training."""

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

from lib.models.common import combine_train_val_sorted, grid_search_refit


def train_forest(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target_column: str = "target",
    *,
    param_grid: dict[str, list[Any]] | None = None,
    scoring: str = "f1",
    verbose: bool = False,
    grid_n_jobs: int = 1,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Fit a random forest. See ``train_decision_tree`` in ``decision_tree`` for grid search behavior."""

    def build(params: dict[str, Any], fit_data: pd.DataFrame) -> RandomForestClassifier:
        kw = {**kwargs, **params}
        if "class_weight" not in kw:
            kw["class_weight"] = "balanced"
        _ = fit_data
        return RandomForestClassifier(random_state=42, **kw)

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
    if "class_weight" not in kwargs:
        kwargs["class_weight"] = "balanced"
    clf = RandomForestClassifier(random_state=42, **kwargs)
    clf.fit(features, targets)
    return clf
