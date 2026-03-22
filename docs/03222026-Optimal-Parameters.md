# Orion experiment: optimal hyperparameters (grid search)

This note records the **best parameters** from `experiments/orion/experiment.py` for symbol **MARA**, with reference symbols **SPY**, **IBIT**, **WGMI**, **CLSK**, **MSTR**. Grid search used **validation F1** as the selection criterion.

## Experiment context

| Setting | Value |
|--------|--------|
| Date range | 2022-01-01 → 2025-12-31 |
| Take profit | 3.0% |
| Stop loss | 3.0% |
| Trade cost | 0.4% |
| Break-even win rate (precision) | ~56.67% |

Train / validation / test splits were time-ordered (see run log for row counts and date spans).

## Optimal parameters by model

| Model | Best parameters | Best F1 (validation) |
|--------|-----------------|------------------------|
| **Decision Tree** | `max_depth`: 8, `min_samples_leaf`: 100 | 0.3560 |
| **Naive Bayes** | `var_smoothing`: 1e-9 | 0.3577 |
| **K-Nearest Neighbors** | `n_neighbors`: 5, `p`: 2, `weights`: `"uniform"` | 0.2785 |
| **Random Forest** | `max_depth`: None, `min_samples_leaf`: 100, `n_estimators`: 100 | **0.4171** |
| **AdaBoost** | `learning_rate`: 0.5, `n_estimators`: 50 | 0.3947 |
| **XGBoost** | `learning_rate`: 0.05, `max_depth`: 4, `n_estimators`: 150, `subsample`: 1.0 | 0.3865 |

Among this run, **Random Forest** achieved the highest validation F1 (0.4171). KNN had the lowest (0.2785).

## Detail: parameter dictionaries

Use these when reproducing or seeding configs:

- **Decision Tree:** `{ "max_depth": 8, "min_samples_leaf": 100 }`
- **Naive Bayes:** `{ "var_smoothing": 1e-09 }`
- **KNN:** `{ "n_neighbors": 5, "p": 2, "weights": "uniform" }`
- **Random Forest:** `{ "max_depth": null, "min_samples_leaf": 100, "n_estimators": 100 }`
- **AdaBoost:** `{ "learning_rate": 0.5, "n_estimators": 50 }`
- **XGBoost:** `{ "learning_rate": 0.05, "max_depth": 4, "n_estimators": 150, "subsample": 1.0 }`

Test-set trading metrics from the same run are in the experiment console output; they are not repeated here.
