#!/usr/bin/env python3
"""Compute 80-dim fbank features cho mọi cut sets, lưu .lca + .jsonl.gz."""
import argparse
from pathlib import Path
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--fbank-dir", required=True)
    ap.add_argument("--num-jobs", type=int, default=8)
    args = ap.parse_args()

    extractor = Fbank(FbankConfig(num_mel_bins=80, sampling_rate=16000))
    md, fd = Path(args.manifest_dir), Path(args.fbank_dir)
    fd.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "test"]:
        src = md / f"cuts_{split}.jsonl.gz"
        if not src.exists():
            continue
        cuts = CutSet.from_jsonl_lazy(src)
        cuts = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=str(fd / f"feats_{split}"),
            storage_type=LilcomChunkyWriter,
            num_jobs=args.num_jobs,
        )
        cuts.to_jsonl(md / f"cuts_{split}_fbank.jsonl.gz")
        print(f"{split}: {len(cuts)}")

if __name__ == "__main__":
    main()
