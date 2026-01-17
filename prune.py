#!/usr/bin/env python
# clean_sentences.py
#
# Usage:
#   python clean_sentences.py sentifm_brat_sentence_dataset_v3.tsv sentifm_sentences_clean.tsv
#
# Input columns expected (header):
#   doc    sent_id    text    y_any_event    types
#
# Output:
#   Same columns, but with cleaned/normalized text. Rows filtered + deduped.

import argparse
import csv
import re
from collections import Counter
from tqdm import tqdm
from typing import Tuple, Optional

WS_RE = re.compile(r"\s+")
URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
FT_LINK_RE = re.compile(r"\b(ft\.com|www\.ft\.com)\b", re.IGNORECASE)

BYLINE_RE = re.compile(r"^\s*by\s+[A-Z][A-Za-z\.\- ]+(and\s+[A-Z][A-Za-z\.\- ]+)?\s*$", re.IGNORECASE)
ADDL_REPORTING_RE = re.compile(r"^\s*additional reporting by\b", re.IGNORECASE)
SEE_LEX_RE = re.compile(r"^\s*see\s+lex\b", re.IGNORECASE)
DAY_RE = re.compile(r"^\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|yesterday)\s*$",
                    re.IGNORECASE)

ALLCAPS_HEADER_RE = re.compile(r"^[A-Z0-9][A-Z0-9 &/\-,'\.]{2,}$")

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = (
        s.replace("\u2018", "'")
         .replace("\u2019", "'")
         .replace("\u201c", '"')
         .replace("\u201d", '"')
    )
    s = WS_RE.sub(" ", s)
    return s

def is_digit_heavy(text: str) -> bool:
    """
    Flags numeric-heavy lines (tables/prices). This is intentionally conservative.
    """
    if not text:
        return False
    digits = sum(ch.isdigit() for ch in text)
    if digits < 8:
        return False
    ratio = digits / max(len(text), 1)
    return ratio > 0.25

def looks_like_sentence(
    text: str,
    *,
    min_tokens: int = 4,     # <- loosened
    min_chars: int = 12,
    max_tokens: int = 80,
    enable_digit_heavy: bool = True,
) -> Tuple[bool, str]:
    t = normalize_text(text)
    if not t:
        return False, "empty"

    if len(t) < min_chars:
        return False, "too_short"

    if URL_RE.search(t) or EMAIL_RE.search(t) or FT_LINK_RE.search(t):
        return False, "url_or_email"

    if BYLINE_RE.match(t) or ADDL_REPORTING_RE.match(t) or SEE_LEX_RE.match(t) or DAY_RE.match(t):
        return False, "boilerplate"

    # Pure section headers like "LONDON", "BANKS", etc.
    if ALLCAPS_HEADER_RE.match(t) and t == t.upper():
        return False, "allcaps_header"

    if enable_digit_heavy and is_digit_heavy(t):
        return False, "digit_heavy"

    toks = t.split()
    if len(toks) < min_tokens:
        return False, "too_short"
    if len(toks) > max_tokens:
        return False, "too_long"

    return True, "ok"

def find_text_col(header_row, preferred_name="text") -> int:
    if not header_row:
        return 2
    try:
        return header_row.index(preferred_name)
    except ValueError:
        return 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_tsv")
    ap.add_argument("output_tsv")
    ap.add_argument("--min-tokens", type=int, default=4)
    ap.add_argument("--min-chars", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--dedupe-case-sensitive", action="store_true",
                    help="If set, dedupe uses exact string; otherwise uses lowercased normalized string.")
    ap.add_argument("--no-digit-heavy-filter", action="store_true")
    args = ap.parse_args()

    reasons = Counter()
    seen = set()

    with open(args.input_tsv, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.reader(f_in, delimiter="\t")
        rows = list(reader)

    if not rows:
        print("input rows:        0")
        print("after filters:     0")
        print("after dedupe:      0")
        print("kept fraction:     0.000")
        return

    header = rows[0]
    data_rows = rows[1:]

    text_col = find_text_col(header, "text")

    total_input = len(rows)  # include header for your reporting style
    kept_after_filter = 0
    kept_after_dedupe = 0

    with open(args.output_tsv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t", lineterminator="\n")
        writer.writerow(header)

        for row in tqdm(data_rows, unit="rows"):
            if not row or text_col >= len(row):
                reasons["missing_text_col"] += 1
                continue

            text = row[text_col]
            ok, reason = looks_like_sentence(
                text,
                min_tokens=args.min_tokens,
                min_chars=args.min_chars,
                max_tokens=args.max_tokens,
                enable_digit_heavy=not args.no_digit_heavy_filter,
            )
            reasons[reason] += 1
            if not ok:
                continue

            kept_after_filter += 1
            norm = normalize_text(text)
            key = norm if args.dedupe_case_sensitive else norm.lower()
            if key in seen:
                continue
            seen.add(key)

            kept_after_dedupe += 1
            out_row = list(row)
            out_row[text_col] = norm
            writer.writerow(out_row)

    print(f"auto text_col:     {text_col}")
    print(f"input rows:        {total_input}")
    print(f"after filters:     {kept_after_filter}")
    print(f"after dedupe:      {kept_after_dedupe}")
    frac = (kept_after_dedupe / (total_input - 1)) if total_input > 1 else 0.0
    print(f"kept fraction:     {frac:.3f}")

    print("\nfilter reasons (counts):")
    for k, v in reasons.most_common():
        print(f"  {k:18s} {v}")

if __name__ == "__main__":
    main()

