#!/usr/bin/env python3
"""Assemble the single-repository Cosmos 3 LoRA dataset bundle.

Thin wrapper: the implementation lives in ``src/temporal_mcq``.  This shim keeps
the historical ``python scripts/<name>`` invocation working; ``python -m
temporal_mcq <command>`` is the preferred entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from temporal_mcq.bundle import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
