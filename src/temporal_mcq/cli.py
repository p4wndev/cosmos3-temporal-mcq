"""Unified command-line entry point.

Usage::

    python -m temporal_mcq <command> [options]

Commands mirror the pipeline stages and forward to the stage modules, which
each keep their own ``main()`` and argument parser.
"""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "download": "temporal_mcq.download",
    "audit": "temporal_mcq.audit",
    "generate": "temporal_mcq.generate",
    "build": "temporal_mcq.build",
    "validate": "temporal_mcq.validate",
    "reports": "temporal_mcq.reports",
    "sheets": "temporal_mcq.sheets",
    "review-server": "temporal_mcq.review_server",
    "review-summary": "temporal_mcq.review",
    "materialize": "temporal_mcq.materialize",
    "verify-release": "temporal_mcq.materialize",
    "bundle": "temporal_mcq.bundle",
    "verify-bundle": "temporal_mcq.bundle",
    "bundle-report": "temporal_mcq.bundle",
    "publish-report": "temporal_mcq.bundle",
}

USAGE = """usage: python -m temporal_mcq <command> [options]

commands:
  download        fetch the source dataset into data/raw/bridgedata2/
  audit           write reports/dataset_audit.md from the source dataset
  generate        generate candidate MCQ questions from structured captions
  build           split by source video and emit TAO annotations
  validate        run CPU-only dataset and TAO schema validation
  reports         write dataset_summary.json, quality_report.md, split_report.md
  sheets          build contact sheets for human visual review
  review-server   serve the local human visual-verification interface
  review-summary  summarise review decisions and remaining unreviewed items
  materialize     build a versioned dataset release from verified decisions
  verify-release  re-check a materialized release against its inputs
  bundle          assemble the single-repo Cosmos 3 LoRA dataset bundle
  verify-bundle   verify an assembled dataset bundle
  bundle-report   plan the bundle (counts, mapping coverage, sizes)
  publish-report  DRY RUN: list files that would be published

Run `python -m temporal_mcq <command> --help` for stage-specific options.
"""

# Commands whose module exposes a differently named entry point.
ENTRY_POINTS = {
    "review-summary": "summary_main",
    "verify-release": "verify_main",
    "verify-bundle": "verify_main",
    "bundle-report": "report_main",
    "publish-report": "publish_main",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE)
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"unknown command: {command}\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    module = importlib.import_module(module_name)
    entry = getattr(module, ENTRY_POINTS.get(command, "main"))
    sys.argv = [f"temporal_mcq {command}", *rest]
    return int(entry())


if __name__ == "__main__":
    raise SystemExit(main())
