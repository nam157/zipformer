#!/usr/bin/env python3
"""Chuẩn hoá transcript tiếng Việt: NFC, lowercase, bỏ punctuation, ghép khoảng trắng,
chuyển số → chữ. Input TSV: <utt_id>\t<text>\t<wav_path>"""
import argparse, csv, re, unicodedata

try:
    from num2words import num2words
except ImportError:
    num2words = None

PUNCT_RE = re.compile(r"[^\wÀ-ỹà-ỹ\s]", flags=re.UNICODE)
WS_RE = re.compile(r"\s+")
NUM_RE = re.compile(r"\d+")

def num_to_vi(m):
    if num2words is None:
        return m.group(0)
    return num2words(int(m.group(0)), lang="vi")

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = NUM_RE.sub(num_to_vi, text)
    text = PUNCT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as fi, open(args.out, "w", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t")
        for row in csv.reader(fi, delimiter="\t"):
            if len(row) < 3:
                continue
            utt, text, wav = row[0], row[1], row[2]
            norm = normalize(text)
            if not norm:
                continue
            w.writerow([utt, norm, wav])

if __name__ == "__main__":
    main()
