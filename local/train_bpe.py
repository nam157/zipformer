#!/usr/bin/env python3
"""Train SentencePiece BPE tokenizer cho tiếng Việt."""
import argparse
from pathlib import Path
import sentencepiece as spm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--vocab-size", type=int, default=500)
    ap.add_argument("--lang-dir", required=True)
    args = ap.parse_args()

    lang_dir = Path(args.lang_dir); lang_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = lang_dir / "bpe"

    spm.SentencePieceTrainer.train(
        input=args.transcript,
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=-1, eos_id=-1, pad_id=-1, unk_id=2,
        user_defined_symbols=["<blk>", "<sos/eos>"],
        normalization_rule_name="nfkc",
    )
    print(f"Saved {model_prefix}.model / .vocab")

if __name__ == "__main__":
    main()
