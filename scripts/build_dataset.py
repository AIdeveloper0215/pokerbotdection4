"""Convert fetched Poker44 benchmark files into a training dataset.

Reads the per-date release files saved by scripts/fetch_benchmark_data.py
(hands_generator/evaluation_datas/*.txt) and writes a flat list of
{"hands": [...], "is_bot": 0|1} records that the model trainer consumes.

Usage:
    python -m scripts.build_dataset --out dataset.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "hands_generator", "evaluation_datas")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default="dataset.json")
    args = ap.parse_args()
    records = []
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.txt"))):
        try:
            payload = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            continue
        for group in payload.get("chunks", []):
            subs = group.get("chunks", [])
            gts = group.get("groundTruth", [])
            for sub, gt in zip(subs, gts):
                if isinstance(sub, list) and sub:
                    records.append({"hands": sub, "is_bot": int(bool(gt))})
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(records, fh)
    print(f"built {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
