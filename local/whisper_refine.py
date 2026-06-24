#!/usr/bin/env python3
"""Refine transcript bằng Whisper: nếu WER(ref, whisper) <= max-wer thì giữ ref;
nếu lệch lớn, có thể thay bằng whisper output hoặc drop."""
import argparse, csv
from pathlib import Path
import jiwer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--max-wer", type=float, default=0.15)
    ap.add_argument("--replace-wer", type=float, default=0.4,
                    help="0.15..0.4: thay bằng whisper; >0.4: drop")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import whisper
    model = whisper.load_model(args.whisper_model, device=args.device)

    kept, replaced, dropped = 0, 0, 0
    with open(args.tsv, encoding="utf-8") as fi, open(args.out, "w", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t")
        for utt, ref, wav in csv.reader(fi, delimiter="\t"):
            wav_p = Path(args.wav_dir) / wav if not Path(wav).is_absolute() else Path(wav)
            if not wav_p.exists():
                continue
            res = model.transcribe(str(wav_p), language="vi", fp16=(args.device == "cuda"))
            hyp = res["text"].strip().lower()
            try:
                wer = jiwer.wer(ref, hyp)
            except Exception:
                wer = 1.0

            if wer <= args.max_wer:
                w.writerow([utt, ref, str(wav_p)]); kept += 1
            elif wer <= args.replace_wer:
                w.writerow([utt, hyp, str(wav_p)]); replaced += 1
            else:
                dropped += 1
    print(f"kept={kept} replaced={replaced} dropped={dropped}")

if __name__ == "__main__":
    main()
