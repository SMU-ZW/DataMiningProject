# TradingSandbox2

Sandbox for **algorithmic trading strategies**: shared libraries for data and experiments, with tests and tooling.

---

## Quick start

1. **Clone** the repo.
2. **Initialize** the project (creates a virtual environment and installs dependencies):
   - Windows: `bin\init.bat`
   - macOS/Linux: `python3 bin/init.py`
3. **Lint and test:** from the project root, run `bin\ctl.bat verify` (Windows) or `python bin/ctl.py verify` (with venv activated).

See [bin/README.md](bin/README.md) for full details on `init.py`, `ctl.py`, and `ctl.bat`.

---

## Directory layout

| Directory      | Purpose |
|----------------|--------|
| **[bin/](bin/)**       | Scripts: setup (`init.py` / `init.bat`) and control (`ctl.py` / `ctl.bat`) for lint and tests. |
| **[etc/](etc/)**       | Data, configs, and private keys. Data and secrets are git-ignored. |
| **[experiments/](experiments/)** | Custom experiments to find and explore trading strategies. |
| **[lib/](lib/)**       | Shared libraries for data manipulation (stock data, utils) used by experiments. |
| **[tests/](tests/)**   | Unit and integration tests for code in `lib`. |

Each directory has its own README with more detail.
