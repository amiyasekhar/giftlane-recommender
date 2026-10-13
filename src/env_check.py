"""Fail fast if project code is run outside the ``giftlane`` conda environment.

Usage::

    from src.env_check import check_env
    check_env()

Raises ``WrongEnvironmentError`` (a ``RuntimeError``) unless

* ``sys.executable`` lives inside ``/opt/homebrew/Caskroom/miniforge/base/envs/giftlane``, and
* the importable ``pandas`` is exactly version ``3.0.6``.

Set ``GIFTLANE_SKIP_ENV_CHECK=1`` to bypass (e.g. on a CI machine with a different prefix).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

EXPECTED_PREFIX = Path("/opt/homebrew/Caskroom/miniforge/base/envs/giftlane")
EXPECTED_PANDAS = "3.0.6"
MESSAGE = "Wrong Python environment. Run: conda activate giftlane"


class WrongEnvironmentError(RuntimeError):
    pass


def check_env() -> Path:
    """Return the resolved interpreter path if the environment is correct; raise otherwise."""
    if os.environ.get("GIFTLANE_SKIP_ENV_CHECK") == "1":
        return Path(sys.executable)

    problems: list[str] = []

    exe = Path(sys.executable).resolve()
    prefix = EXPECTED_PREFIX.resolve()
    if exe != prefix and prefix not in exe.parents:
        problems.append(f"python is {sys.executable}\n      expected one inside {EXPECTED_PREFIX}")

    try:
        import pandas  # noqa: PLC0415

        if pandas.__version__ != EXPECTED_PANDAS:
            problems.append(f"pandas is {pandas.__version__}, expected {EXPECTED_PANDAS}")
    except ImportError:
        problems.append(f"pandas is not installed, expected {EXPECTED_PANDAS}")

    if problems:
        details = "\n".join(f"  - {p}" for p in problems)
        raise WrongEnvironmentError(f"{MESSAGE}\n{details}")
    return exe


if __name__ == "__main__":
    print("OK:", check_env())
