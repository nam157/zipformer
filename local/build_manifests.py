#!/usr/bin/env python3
"""Build Lhotse manifests (train/dev/test) từ TSV refined."""
import argparse, csv, random
from pathlib import Path
from lhotse import Recording, SupervisionSegment, RecordingSet, SupervisionSet, CutSet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dev-ratio", type=float, default=0.01)
    ap.add_argument("--test-ratio", type=float, default=0.01)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.reader(open(args.tsv, encoding="utf-8"), delimiter="\t"))
    random.Random(args.seed).shuffle(rows)

    recs, sups = [], []
    for utt, text, wav in rows:
        wav_p = Path(wav)
        if not wav_p.is_absolute():
            wav_p = Path(args.wav_dir) / wav
        try:
            r = Recording.from_file(wav_p, recording_id=utt)
        except Exception:
            continue
        if not (args.min_dur <= r.duration <= args.max_dur):
            continue
        recs.append(r)
        sups.append(SupervisionSegment(
            id=utt, recording_id=utt, start=0.0, duration=r.duration,
            channel=0, text=text, language="Vietnamese"))

    rec_set, sup_set = RecordingSet.from_recordings(recs), SupervisionSet.from_segments(sups)
    cuts = CutSet.from_manifests(recordings=rec_set, supervisions=sup_set).trim_to_supervisions()

    n = len(cuts)
    n_dev = max(1, int(n * args.dev_ratio))
    n_test = max(1, int(n * args.test_ratio))
    cuts_list = list(cuts)
    dev_cuts = CutSet.from_cuts(cuts_list[:n_dev])
    test_cuts = CutSet.from_cuts(cuts_list[n_dev:n_dev + n_test])
    train_cuts = CutSet.from_cuts(cuts_list[n_dev + n_test:])

    train_cuts.to_jsonl(out / "cuts_train.jsonl.gz")
    dev_cuts.to_jsonl(out / "cuts_dev.jsonl.gz")
    test_cuts.to_jsonl(out / "cuts_test.jsonl.gz")
    print(f"train={len(train_cuts)} dev={len(dev_cuts)} test={len(test_cuts)}")

if __name__ == "__main__":
    main()
