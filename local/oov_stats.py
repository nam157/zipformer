#!/usr/bin/env python3
"""Phân tích token tail (rare/OOV-like) để dùng cho oversample augment."""
import argparse, json
from collections import Counter
import sentencepiece as spm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpe-model", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-count", type=int, default=5)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.bpe_model)
    c = Counter()
    with open(args.transcript, encoding="utf-8") as f:
        for line in f:
            c.update(sp.encode(line.strip(), out_type=str))

    rare = sorted([t for t, n in c.items() if n < args.min_count])
    json.dump({
        "min_count": args.min_count,
        "rare_tokens": rare,
        "total_tokens": len(c),
        "rare_ratio": len(rare) / max(1, len(c)),
    }, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"rare={len(rare)}/{len(c)}")

if __name__ == "__main__":
    main()
