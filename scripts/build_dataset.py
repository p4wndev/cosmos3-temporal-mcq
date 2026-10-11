#!/usr/bin/env python3
"""Split by source video and emit TAO annotations.

Thin wrapper: the implementation lives in ``src/temporal_mcq``.  This shim keeps
the historical ``python scripts/<name>`` invocation working; ``python -m
temporal_mcq <command>`` is the preferred entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``src/`` importable when the package has not been installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from temporal_mcq.build import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
