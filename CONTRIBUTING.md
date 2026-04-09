# Contributing to tandem-solar

Thanks for your interest! This guide will get you onboarded as a real
collaborator.

## Local setup

```bash
git clone https://github.com/<your-username>/tandem-solar.git
cd tandem-solar

python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# Install in editable mode plus test deps
pip install -e .
pip install -r requirements.txt
pip install pytest
```

Python 3.10+ is required (PEP 604 type hints). Core scientific dependencies
are `numpy`, `scipy`, and `matplotlib`.

## Running

`tandem-solar` is primarily a library imported by other code:

```python
from tandem import SolarCell, TandemModule, CTMLossAnalyser
cell = SolarCell()
print(cell.mpp())
```

A minimal CLI is provided for self-tests:

```bash
# Self-test (good for CI smoke checks)
python -m tandem --health-check

# Verbose / debug logging
python -m tandem --health-check -vv
```

The `examples/` directory shows end-to-end usage and `benchmarks/` contains
performance scripts.

## Tests

```bash
python -m pytest tests/ -v
```

All tests must pass before opening a PR. New features should add tests.

## Code conventions

- **Use the `logging` module**, not `print`, for diagnostics. Library modules
  attach a `NullHandler` and never call `basicConfig`. Pick the right level:
  `DEBUG` (internals), `INFO` (milestones), `WARNING` (recoverable), `ERROR`
  (failures).
- **Validate inputs at public-API boundaries.** Raise `ValueError`/`TypeError`
  with messages that say *what* was wrong and *what* was received.
- Keep functions documented with NumPy-style docstrings (see existing code).
- No new third-party runtime dependencies without discussion.

## Pull request process

1. **Open an issue first** for anything larger than a small fix.
2. **Branch from `main`** with a descriptive name.
3. **Make focused commits** — one logical change per PR.
4. **Run tests + `python -m tandem --health-check`** before pushing.
5. **Open the PR against `main`** and include:
   - What changed and why
   - How you tested it
   - Any follow-ups
6. **Address review feedback** with additional commits (no force-push during
   review unless asked).
7. A maintainer will merge once review is resolved.

## Reporting bugs

Open a GitHub issue with: expected vs actual behaviour, minimal reproducer,
Python version, OS, and `-vv` debug logs.
