"""Fail fast if project code is run outside the ``giftlane`` conda environment.

Usage::

    from src.env_check import check_env
    check_env()

Raises ``WrongEnvironmentError`` (a ``RuntimeError``) unless

* the running interpreter is Python 3.11 (any patch release), and
* the importable ``pandas`` is exactly version ``3.0.6``.

The check is deliberately path-free so that a fresh ``conda create -n giftlane python=3.11`` plus
``python -m pip install -r requirements.txt`` passes on any machine.

Set ``GIFTLANE_SKIP_ENV_CHECK=1`` to bypass (e.g. on a CI machine with a different setup).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

EXPECTED_PYTHON = (3, 11)
EXPECTED_PANDAS = "3.0.6"
MESSAGE = "Wrong Python environment. Run: conda activate giftlane"


class WrongEnvironmentError(RuntimeError):
    pass


def check_env() -> Path:
    """Return the interpreter path if the environment is correct; raise otherwise."""
    if os.environ.get("GIFTLANE_SKIP_ENV_CHECK") == "1":
        return Path(sys.executable)

    problems: list[str] = []

    actual_python = sys.version_info[:2]
    if actual_python != EXPECTED_PYTHON:
        problems.append(
            f"python is {actual_python[0]}.{actual_python[1]} ({sys.executable}), "
            f"expected {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}"
        )

    try:
        import pandas  # noqa: PLC0415

        if pandas.__version__ != EXPECTED_PANDAS:
            problems.append(f"pandas is {pandas.__version__}, expected {EXPECTED_PANDAS}")
    except ImportError:
        problems.append(f"pandas is not installed, expected {EXPECTED_PANDAS}")

    if problems:
        details = "\n".join(f"  - {p}" for p in problems)
        raise WrongEnvironmentError(f"{MESSAGE}\n{details}")
    return Path(sys.executable)


if __name__ == "__main__":
    print("OK:", check_env())
