#!/usr/bin/env python
# make_brat_sentence_tsv_spacy_v3ish.py
#
# Aggressive-ish: spaCy sentences + optional sub-splitting on ; : — and newlines.
#
# Usage:
#   python make_brat_sentence_tsv_spacy_v3ish.py \
#     --input_dir osfstorage/bratdata/bratannotationfiles \
#     --output_tsv sentifm_brat_sentence_dataset_v3_spacyish.tsv
#
# Requires:
#   pip install spacy
#   python -m spacy download en_core_web_sm

import argparse
import csv
import os
import re
from tqdm import tqdm
from typing import List, Tuple, Set

import spacy

EVENT_TYPES = {
    "Profit",
    "Turnover",
    "SalesVolume",
    "ShareRepurchase",
    "Debt",
    "QuarterlyResults",
    "TargetPrice",
    "BuyRating",
    "Dividend",
    "MergerAcquisition",
}

WS_RE = re.compile(r"\s+")
# Split points that often separate “sentence-like” units in this corpus
SPLIT_RE = re.compile(r"(?:(?<=;)\s+|(?<=:)\s+|(?<=\u2014)\s+|(?<=\u2013)\s+|(?<=--)\s+|\n+)")


def normalize_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return (a0 < b1) and (b0 < a1)


def parse_brat_ann(path: str) -> List[Tuple[str, int, int]]:
    """
    Return list of (type, start, end) for event entities only.
    Supports discontinuous spans: start1 end1;start2 end2
    """
    spans: List[Tuple[str, int, int]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            spec = parts[1]
            first_space = spec.find(" ")
            if first_space <= 0:
                continue
            ent_type = spec[:first_space]
            if ent_type not in EVENT_TYPES:
                continue
            offsets_str = spec[first_space + 1 :].strip()
            for chunk in offsets_str.split(";"):
                chunk = chunk.strip()
                nums = chunk.split()
                if len(nums) != 2:
                    continue
                try:
                    start = int(nums[0])
                    end = int(nums[1])
                except ValueError:
                    continue
                if end > start:
                    spans.append((ent_type, start, end))
    return spans


def iter_pairs(input_dir: str):
    for fn in os.listdir(input_dir):
        if not fn.endswith(".txt"):
            continue
        base = fn[:-4]
        txt_path = os.path.join(input_dir, base + ".txt")
        ann_path = os.path.join(input_dir, base + ".ann")
        if os.path.isfile(txt_path) and os.path.isfile(ann_path):
            yield base, txt_path, ann_path


def split_sentence_like(text: str, min_len: int, max_chars: int, extra_split: bool) -> List[str]:
    """
    Start from a sentence string (already spaCy segmented),
    then optionally split into smaller “sentence-like” pieces.
    """
    t = text
    if not extra_split:
        t2 = normalize_ws(t)
        return [t2] if (len(t2) >= min_len and len(t2) <= max_chars) else []

    parts = [p for p in SPLIT_RE.split(t) if p and not p.isspace()]
    out: List[str] = []
    for p in parts:
        p2 = normalize_ws(p)
        if len(p2) < min_len:
            continue
        if len(p2) > max_chars:
            # If still huge, try a last-resort split on comma+space
            # (keeps it from nuking everything; conservative)
            subparts = [normalize_ws(x) for x in p2.split(", ") if x]
            for sp in subparts:
                if min_len <= len(sp) <= max_chars:
                    out.append(sp)
            continue
        out.append(p2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_tsv", required=True)
    ap.add_argument("--model", default="en_core_web_sm")
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--max_chars", type=int, default=500)
    ap.add_argument("--extra_split", action="store_true",
                    help="Split spaCy sentences further on ; : dashes and newlines (more v3-like)")
    args = ap.parse_args()

    nlp = spacy.load(args.model)

    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        raise RuntimeError(
            "spaCy pipeline has no 'parser' or 'senter'. Use en_core_web_sm (or similar)."
        )

    print(f"spaCy version: {spacy.__version__}")
    print(f"pipeline: {nlp.pipe_names}")

    pairs = sorted(list(iter_pairs(args.input_dir)))
    print(f"paired docs: {len(pairs)}")

    wrote = 0
    with open(args.output_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["doc", "sent_id", "text", "y_any_event", "types"])

        for doc_id, txt_path, ann_path in tqdm(pairs, unit="pairs"):
            raw = read_text(txt_path)
            ev_spans = parse_brat_ann(ann_path)
            doc = nlp(raw)

            sent_id = 0
            for sent in doc.sents:
                s0, s1 = sent.start_char, sent.end_char

                # Which event types overlap the ORIGINAL spaCy sentence span?
                # (Evidence must appear within that sentence; sub-splitting is text-only.)
                hit_types: Set[str] = set()
                for t, e0, e1 in ev_spans:
                    if overlaps(s0, s1, e0, e1):
                        hit_types.add(t)

                y = 1 if hit_types else 0
                types_str = "|".join(sorted(hit_types))

                # Now split the sentence text into more units (v3-ish)
                pieces = split_sentence_like(sent.text, args.min_len, args.max_chars, args.extra_split)
                for piece in pieces:
                    w.writerow([doc_id, sent_id, piece, y, types_str])
                    wrote += 1
                    sent_id += 1

    print(f"wrote rows: {wrote}")
    print(f"output: {args.output_tsv}")


if __name__ == "__main__":
    main()

