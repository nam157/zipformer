#!/usr/bin/env python3
"""Full evaluation: WER/CER per subset + RTF benchmark trên CPU.

  python zipformer/eval.py \
    --exp-dir exp/vi_zipformer --epoch 30 --avg 10 \
    --bpe-model data/lang_bpe_500/bpe.model \
    --manifest-dir data/manifests --fbank-dir data/fbank \
    --chunk-sizes 16,32,64 --bench-rtf 1
"""
import argparse, json, time
from pathlib import Path
import torch, sentencepiece as spm, jiwer, numpy as np

from asr_datamodule import VietnameseAsrDataModule
from model import VietZipformerRNNT
from decode import greedy_search
from icefall.checkpoint import average_checkpoints, load_checkpoint


def evaluate(model, dl, sp, device):
    refs, hyps, audio_sec, infer_sec = [], [], 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for batch in dl:
            feats = batch["inputs"].to(device)
            feat_lens = batch["supervisions"]["num_frames"].to(device)
            cuts = batch["supervisions"]["cut"]
            t0 = time.time()
            x, x_lens = model.encoder_embed(feats, feat_lens)
            x = x.permute(1, 0, 2)
            enc_out, enc_lens = model.encoder(x, x_lens)
            enc_out = enc_out.permute(1, 0, 2)
            hyp_ids = greedy_search(model, enc_out, enc_lens, blank_id=0)
            infer_sec += time.time() - t0
            for c, ids in zip(cuts, hyp_ids):
                refs.append(c.supervisions[0].text)
                hyps.append(sp.decode(ids))
                audio_sec += c.duration
    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        "rtf": infer_sec / max(1e-9, audio_sec),
        "n": len(refs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--fbank-dir", required=True)
    ap.add_argument("--bpe-model", required=True)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--avg", type=int, default=1)
    ap.add_argument("--chunk-sizes", default="16,32,64")
    ap.add_argument("--bench-rtf", type=int, default=1)
    ap.add_argument("--max-duration", type=float, default=300)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.bpe_model)
    device_eval = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chunks = [int(x) for x in args.chunk_sizes.split(",")]
    results = {}

    for cs in chunks:
        model = VietZipformerRNNT(
            vocab_size=sp.get_piece_size(), blank_id=0,
            chunk_size=(cs,), left_context_frames=(cs * 4,),
        )
        ckpts = [Path(args.exp_dir) / f"epoch-{e}.pt"
                 for e in range(args.epoch - args.avg + 1, args.epoch + 1)]
        if args.avg > 1:
            model.load_state_dict(average_checkpoints(ckpts, device="cpu"), strict=False)
        else:
            load_checkpoint(ckpts[-1], model=model)

        # WER/CER chạy trên GPU (nếu có) cho nhanh
        model.to(device_eval)
        dm = VietnameseAsrDataModule(args.manifest_dir, args.fbank_dir,
                                     bpe_model=args.bpe_model,
                                     max_duration=args.max_duration)
        r = {}
        for split, dl in [("dev", dm.dev_dataloader()), ("test", dm.test_dataloader())]:
            r[split] = evaluate(model, dl, sp, device_eval)
            r[split].pop("rtf")  # RTF GPU không có ý nghĩa, đo riêng dưới

        # RTF benchmark single-thread CPU
        if args.bench_rtf:
            model_cpu = model.to("cpu")
            torch.set_num_threads(1)
            dm_cpu = VietnameseAsrDataModule(args.manifest_dir, args.fbank_dir,
                                             bpe_model=args.bpe_model,
                                             max_duration=60, num_workers=0)
            r["rtf_cpu_1thread"] = evaluate(model_cpu, dm_cpu.test_dataloader(),
                                            sp, torch.device("cpu"))["rtf"]
        results[f"chunk_{cs}"] = r
        print(f"chunk={cs}: {json.dumps(r, indent=2, ensure_ascii=False)}")

    out = Path(args.exp_dir) / f"eval-epoch{args.epoch}-avg{args.avg}.json"
    json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
