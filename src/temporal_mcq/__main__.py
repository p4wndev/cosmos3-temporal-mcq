"""Allow ``python -m temporal_mcq`` to run the CLI dispatcher."""
from __future__ import annotations

from temporal_mcq.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
