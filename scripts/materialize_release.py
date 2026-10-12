#!/usr/bin/env python3
"""Materialize a versioned dataset release from confirmed review decisions.

Thin wrapper: the implementation lives in ``src/temporal_mcq``.  This shim keeps
the historical ``python scripts/<name>`` invocation working; ``python -m
temporal_mcq <command>`` is the preferred entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``src/`` importable when the package has not been installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from temporal_mcq.materialize import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
