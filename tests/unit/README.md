# unit

**Unit tests** for code in `lib`.

Unit tests must be **fast** and must **not rely on outside entities** (no network calls, no real broker APIs, no external services). Use mocks, fakes, or in-memory data so the suite can run quickly and offline.

**Naming:** Mirror nested packages in `lib` with a directory of the same name (e.g. `lib/stock` → `unit/stock/`). Name each test file `test_<the module's name>.py` (e.g. tests for `lib/stock/data_cleaner.py` go in `unit/stock/test_data_cleaner.py`). For `lib/models/`, use one file per module (e.g. `lib/models/decision_tree.py` → `unit/models/test_decision_tree.py`) plus shared fixtures in `unit/models/tiny_frames.py` as needed.
