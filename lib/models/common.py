"""Shared helpers for sklearn / XGBoost training with optional grid search."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

EstimatorT = TypeVar("EstimatorT")


def combine_train_val_sorted(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate train and validation frames and sort by time (for refit after tuning)."""
    return pd.concat([train_df, validation_df]).sort_index(level="timestamp")


def validation_score(
    y_val: pd.Series,
    y_pred: np.ndarray,
    estimator: Any,
    x_val: pd.DataFrame,
    scoring: str,
) -> float:
    """Return validation metric: ``f1`` (hard predictions) or ``roc_auc`` (``predict_proba``)."""
    if scoring == "f1":
        return float(f1_score(y_val, y_pred, zero_division=0))
    if scoring == "roc_auc":
        try:
            y_proba = estimator.predict_proba(x_val)[:, 1]
            return float(roc_auc_score(y_val, y_proba))
        except ValueError:
            return 0.0
    raise ValueError(f"unknown scoring: {scoring!r} (use 'f1' or 'roc_auc')")


def xgboost_scale_pos_weight(y: pd.Series) -> float:
    """``n_negative / n_positive`` for binary ``y``; 1.0 if there are no positives."""
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return 1.0
    return float(int((y == 0).sum()) / n_pos)


def _effective_grid_n_jobs(grid_n_jobs: int, n_tasks: int) -> int:
    """Map ``grid_n_jobs`` to a thread count (sklearn-style: ``-1`` = all CPUs)."""
    if n_tasks <= 1:
        return 1
    if grid_n_jobs == 0:
        return 1
    if grid_n_jobs < 0:
        cpus = os.cpu_count() or 1
        want = cpus + 1 + grid_n_jobs
        want = max(1, want)
        return min(n_tasks, want)
    return min(n_tasks, grid_n_jobs)


def grid_search_refit(
    build: Callable[[dict[str, Any], pd.DataFrame], EstimatorT],
    param_grid: dict[str, list[Any]],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target_column: str,
    scoring: str,
    verbose: bool,
    *,
    grid_n_jobs: int = 1,
) -> EstimatorT:
    """Fit each grid point on ``train_df``, score on ``validation_df``, refit best on train+val.

    When ``grid_n_jobs`` > 1 (or ``-1`` for all CPUs), candidate parameter sets are evaluated in
    parallel using threads. Estimator fitting for sklearn and XGBoost mostly runs outside the
    GIL, so threads usually speed up the search without pickling data to child processes. Pass
    estimator parallelism via ``build`` / model kwargs (e.g. ``n_jobs`` on the classifier) and
    tune ``grid_n_jobs`` separately to avoid oversubscribing CPU.
    """
    if not param_grid:
        raise ValueError("param_grid must be non-empty for grid search")
    x_fit = train_df.drop(columns=[target_column])
    y_fit = train_df[target_column]
    x_val = validation_df.drop(columns=[target_column])
    y_val = validation_df[target_column]

    grid_list = list(ParameterGrid(param_grid))
    workers = _effective_grid_n_jobs(grid_n_jobs, len(grid_list))

    def eval_indexed(i: int) -> tuple[int, dict[str, Any], float]:
        params = grid_list[i]
        est = build(params, train_df)
        est.fit(x_fit, y_fit)
        y_pred = est.predict(x_val)
        score = validation_score(y_val, y_pred, est, x_val, scoring)
        return i, params, score

    if workers <= 1:
        scores = [eval_indexed(i)[1:] for i in range(len(grid_list))]
    else:
        ordered: list[tuple[dict[str, Any], float] | None] = [None] * len(grid_list)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(eval_indexed, i): i for i in range(len(grid_list))}
            for fut in as_completed(futures):
                i, params, score = fut.result()
                ordered[i] = (params, score)
        scores = [t for t in ordered if t is not None]

    best_score = -np.inf
    best_params: dict[str, Any] | None = None
    for params, score in scores:
        if score > best_score:
            best_score = score
            best_params = params
    if best_params is None:
        raise RuntimeError("grid search found no valid parameter set")
    if verbose:
        print(f"  best params: {best_params} | best {scoring} (val): {best_score:.4f}")
    refit_df = combine_train_val_sorted(train_df, validation_df)
    final = build(best_params, refit_df)
    x_full = refit_df.drop(columns=[target_column])
    y_full = refit_df[target_column]
    final.fit(x_full, y_full)
    return final
