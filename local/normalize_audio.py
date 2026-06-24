#!/usr/bin/env python3
"""Resample mọi audio về 16kHz mono wav, dùng ffmpeg + multiprocessing."""
import argparse, os, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}

def convert(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-ar", "16000", "-ac", "1",
         "-af", "loudnorm=I=-23:LRA=7:TP=-2",
         str(dst)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--dst-dir", required=True)
    ap.add_argument("--num-jobs", type=int, default=8)
    args = ap.parse_args()

    src_root, dst_root = Path(args.src_dir), Path(args.dst_dir)
    files = [p for p in src_root.rglob("*") if p.suffix.lower() in EXTS]
    print(f"Found {len(files)} files")

    with ProcessPoolExecutor(args.num_jobs) as ex:
        futs = []
        for p in files:
            rel = p.relative_to(src_root).with_suffix(".wav")
            futs.append(ex.submit(convert, p, dst_root / rel))
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 200 == 0:
                print(f"  {i}/{len(files)}")

if __name__ == "__main__":
    main()
