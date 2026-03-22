"""XGBoost classifier training."""

from typing import Any

import pandas as pd
from xgboost import XGBClassifier  # type: ignore[import-untyped]

from lib.models.common import combine_train_val_sorted, grid_search_refit, xgboost_scale_pos_weight


def train_xgboost(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target_column: str = "target",
    *,
    param_grid: dict[str, list[Any]] | None = None,
    scoring: str = "f1",
    verbose: bool = False,
    grid_n_jobs: int = 1,
    **kwargs: Any,
) -> XGBClassifier:
    """Fit an XGBoost classifier.

    When ``scale_pos_weight`` is omitted, it is set from the labels of the dataframe passed to
    each ``build`` (train-only during grid search; train+val on final refit).

    See ``train_decision_tree`` in ``decision_tree`` for grid search vs single-fit behavior.
    """

    def build(params: dict[str, Any], fit_data: pd.DataFrame) -> XGBClassifier:
        kw = {**kwargs, **params}
        if "scale_pos_weight" not in kw:
            kw["scale_pos_weight"] = xgboost_scale_pos_weight(fit_data[target_column])
        return XGBClassifier(random_state=42, **kw)

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
    if "scale_pos_weight" not in kwargs:
        kwargs["scale_pos_weight"] = xgboost_scale_pos_weight(targets)
    clf = XGBClassifier(random_state=42, **kwargs)
    clf.fit(features, targets)
    return clf
