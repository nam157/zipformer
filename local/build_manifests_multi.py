#!/usr/bin/env python3
"""Build manifests cho multi-dataset, có thông tin source + tier để:
   - Stage A: chỉ lấy tier=clean
   - Stage B: lấy all, oversample clean ×2
   - Per-source dev/test để đánh giá riêng từng domain
TSV input: utt_id\ttext\twav_path\tsource\ttier
"""
import argparse, csv, random
from collections import defaultdict
from pathlib import Path
from lhotse import Recording, SupervisionSegment, RecordingSet, SupervisionSet, CutSet

def load_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t"):
            if len(r) >= 5:
                rows.append(r[:5])
    return rows

def to_cutset(rows, min_dur, max_dur):
    recs, sups = [], []
    for uid, text, wav, src, tier in rows:
        try:
            r = Recording.from_file(wav, recording_id=uid)
        except Exception:
            continue
        if not (min_dur <= r.duration <= max_dur): continue
        recs.append(r)
        sups.append(SupervisionSegment(
            id=uid, recording_id=uid, start=0.0, duration=r.duration,
            channel=0, text=text, language="Vietnamese",
            custom={"source": src, "tier": tier},
        ))
    cuts = CutSet.from_manifests(
        recordings=RecordingSet.from_recordings(recs),
        supervisions=SupervisionSet.from_segments(sups),
    ).trim_to_supervisions()
    return cuts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--dev-per-source", type=int, default=500)
    ap.add_argument("--test-per-source", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = load_tsv(args.tsv)
    by_src = defaultdict(list)
    for r in rows: by_src[r[3]].append(r)

    rng = random.Random(args.seed)
    train_rows, dev_by_src, test_by_src = [], {}, {}
    for src, rs in by_src.items():
        rng.shuffle(rs)
        d = rs[:args.dev_per_source]
        t = rs[args.dev_per_source:args.dev_per_source + args.test_per_source]
        tr = rs[args.dev_per_source + args.test_per_source:]
        dev_by_src[src] = d; test_by_src[src] = t; train_rows += tr
        print(f"{src:15s} train={len(tr):>7d} dev={len(d)} test={len(t)}")

    # Train: 2 manifest — clean-only và full
    clean_rows = [r for r in train_rows if r[4] == "clean"]
    print(f"train_clean={len(clean_rows)}  train_full={len(train_rows)}")

    to_cutset(clean_rows, args.min_dur, args.max_dur).to_jsonl(
        out / "cuts_train_clean.jsonl.gz")
    to_cutset(train_rows, args.min_dur, args.max_dur).to_jsonl(
        out / "cuts_train_full.jsonl.gz")

    # Dev / test per-source
    for src, rs in dev_by_src.items():
        to_cutset(rs, args.min_dur, args.max_dur).to_jsonl(
            out / f"cuts_dev_{src}.jsonl.gz")
    for src, rs in test_by_src.items():
        to_cutset(rs, args.min_dur, args.max_dur).to_jsonl(
            out / f"cuts_test_{src}.jsonl.gz")

    # Aggregate dev/test
    all_dev = [r for rs in dev_by_src.values() for r in rs]
    all_test = [r for rs in test_by_src.values() for r in rs]
    to_cutset(all_dev, args.min_dur, args.max_dur).to_jsonl(
        out / "cuts_dev.jsonl.gz")
    to_cutset(all_test, args.min_dur, args.max_dur).to_jsonl(
        out / "cuts_test.jsonl.gz")

if __name__ == "__main__":
    main()
