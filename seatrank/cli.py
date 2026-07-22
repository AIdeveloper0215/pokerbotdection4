"""Command-line trainer: python -m seatrank.cli --data merged.json --out model.joblib"""
from __future__ import annotations

import argparse
import json

from .forest import fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="seatrank_forest.joblib")
    args = ap.parse_args()
    records = json.load(open(args.data, "r", encoding="utf-8"))
    fit(records, args.out)
    print(f"trained {len(records)} chunks -> {args.out}")


if __name__ == "__main__":
    main()
