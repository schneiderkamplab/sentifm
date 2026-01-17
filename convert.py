#!/usr/bin/env python
"""
tsv_to_jsonl.py

Convert a TSV file to JSONL (one JSON object per line).

Usage:
  python tsv_to_jsonl.py input.tsv output.jsonl
  python tsv_to_jsonl.py input.tsv output.jsonl --label-col y_any_event
  python tsv_to_jsonl.py input.tsv output.jsonl --text-col text --drop-cols doc,sent_id
"""

import argparse
import csv
import json
from tqdm import tqdm
from typing import Dict, Any, Optional, Set


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_tsv")
    ap.add_argument("output_jsonl")
    ap.add_argument("--text-col", default="text", help="Column name for the text field.")
    ap.add_argument("--label-col", default=None, help="Optional label column name (e.g., y_any_event).")
    ap.add_argument("--drop-cols", default="", help="Comma-separated columns to drop (e.g., doc,sent_id).")
    ap.add_argument("--keep-empty", action="store_true", help="Keep rows with empty text.")
    return ap.parse_args()


def main():
    args = parse_args()
    drop: Set[str] = {c.strip() for c in args.drop_cols.split(",") if c.strip()}

    with open(args.input_tsv, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit("No header found in TSV.")

        if args.text_col not in reader.fieldnames:
            raise SystemExit(f"--text-col '{args.text_col}' not in header: {reader.fieldnames}")

        if args.label_col and args.label_col not in reader.fieldnames:
            raise SystemExit(f"--label-col '{args.label_col}' not in header: {reader.fieldnames}")

        with open(args.output_jsonl, "w", encoding="utf-8") as fout:
            for row in tqdm(reader, unit="lines"):
                text = (row.get(args.text_col) or "").strip()
                if not args.keep_empty and not text:
                    continue

                obj: Dict[str, Any] = {}

                # Standardize keys if desired: always include "text"
                obj["text"] = text

                # Optional: include label as int if possible
                if args.label_col:
                    raw = (row.get(args.label_col) or "").strip()
                    try:
                        obj["label"] = int(raw)
                    except ValueError:
                        # fall back to raw string if it isn't an int
                        obj["label"] = raw

                # Include remaining columns (minus dropped/text/label)
                for k, v in row.items():
                    if k in drop or k == args.text_col or (args.label_col and k == args.label_col):
                        continue
                    obj[k] = v

                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

