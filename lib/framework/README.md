# Framework

Broker-agnostic trading abstractions for strategies, orders, positions, and execution. The same interfaces are used for **backtest** (via `lib.backtest`) and will support **paper/live** later by swapping implementations of the data feed, broker, and clock.

## Modules

### orders (`orders.py`)

Core types for sending and recording executions:

- **Order** — A request to trade: `symbol`, `side` (buy/sell), `qty`, `order_type` (market, limit, stop, stop_limit), optional `limit_price`/`stop_price`, optional `id`. Validates qty > 0 and that limit/stop orders have the required prices.
- **Fill** — An execution: `order_id`, `symbol`, `side`, `price`, `qty`, `timestamp`. Used by the portfolio to update state and kept in `trade_history` for later metrics.
- **OrderSide** — Enum: `BUY`, `SELL` (subclasses `str` so you can pass `"buy"`/`"sell"` or use the enum).
- **OrderType** — Enum: `MARKET`, `LIMIT`, `STOP`, `STOP_LIMIT`.

### portfolio (`portfolio.py`)

Portfolio state and valuation:

- **Position** — Single symbol: `quantity`, `cost_basis`, and `avg_price` (derived). Used for cost-basis and P&L.
- **Portfolio** — Holds `cash`, a dict of `positions` by symbol, and `trade_history` (list of `Fill`). Call `apply_fill(fill)` to update cash and positions from broker fills (no short selling). Call `equity(snapshot)` for total equity; pass a DataFrame with MultiIndex (symbol, timestamp) and `close` column for mark-to-market, or omit for book value (cost basis).

### strategy (`strategy.py`)

- **Strategy** — A `typing.Protocol`. Implement `next(current_time, market_snapshot, portfolio) -> list[Order]`. Called each bar (or tick in live) with the current time, a DataFrame of bar(s), and the current portfolio; return the orders to submit. The same strategy implementation can run in backtest or paper/live.

### data_feed (`data_feed.py`)

- **DataFeed** — Abstract base. Implement `get_bars(current_time, symbol=None) -> DataFrame`. Backtest uses a DataFrame-backed implementation; paper/live can wrap a broker or API to return the latest bar(s). Data shape matches `lib.stock` (MultiIndex symbol/timestamp, OHLCV columns).

### broker (`broker.py`)

- **Broker** — Abstract base. Implement `submit(order)` and `get_fills() -> list[Fill]`. Backtest uses a simulated broker that fills at bar close; paper/live will use Alpaca (or another broker). After each step, call `get_fills()` and apply fills to the portfolio.

### clock (`clock.py`)

- **Clock** — Abstract base. Implement `current_time -> datetime`. In backtest this is the current bar time; in live it is the real clock. Used to drive the main loop and pass time into the strategy and data feed.

## Usage

Import from the package:

```python
from lib.framework import (
    Order, Fill, OrderSide, OrderType,
    Portfolio, Position,
    Strategy, DataFeed, Broker, Clock,
)
```

Implement `Strategy`, and (for backtest) provide concrete `DataFeed`, `Broker`, and `Clock`; the backtest engine wires them together and calls `strategy.next()` each step, submits orders, applies fills to the portfolio, and advances the clock.
