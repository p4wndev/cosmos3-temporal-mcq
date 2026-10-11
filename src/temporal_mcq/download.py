#!/usr/bin/env python3
"""Download the NVIDIA BridgeData2 synthetic-caption subset into ./data/raw/bridgedata2/.

Uses HF_TOKEN only if present in the environment (this dataset is public and
ungated, so no token is required). Never prints credential values.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = "nvidia/BridgeData2-Subset-Synthetic-Captions"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "bridgedata2"

def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    print(f"[download] repo={REPO}")
    print(f"[download] target={RAW}")
    print(f"[download] HF_TOKEN present={bool(token)} (value never printed)")
    start = time.time()
    path = snapshot_download(
        repo_id=REPO, repo_type="dataset", local_dir=str(RAW),
        token=token, max_workers=8,
    )
    print(f"[download] complete in {time.time()-start:.1f}s -> {path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
