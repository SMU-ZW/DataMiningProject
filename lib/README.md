# lib

This folder holds **common libraries** for manipulating data that the **experiments** can use.

---

## Subdirectories

| Directory   | Purpose |
|------------|---------|
| **`stock/`** | Classes and code for pulling and cleaning stock data. |
| **`utils/`** | Generic utility code (e.g. conversions, helpers). |
| **`common/`** | Shared feature-engineering helpers on DataFrames (used by experiments and data-mining workflows). |
| **`models/`** | Reusable ML training helpers: scikit-learn and XGBoost wrappers, grid search, train/validation/test flow (fit scaler on train, refit after tuning, etc.). |
| **`framework/`** | Broker-agnostic abstractions: orders, portfolio, strategy, data feed, broker, clock. Same interfaces for backtest and (later) paper/live. |
| **`backtest/`** | Backtest engine that runs strategies over historical bars using the framework and `lib.stock` data shape. |

Import from `lib` in your experiment scripts (e.g. `from lib.stock import ...`, `from lib.models import ...`, `from lib.backtest import run`, `from lib.framework import ...`). Backtest uses the framework and stock data; paper/live will reuse the framework with different data-feed and broker implementations.
