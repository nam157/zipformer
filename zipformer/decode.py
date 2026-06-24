#!/usr/bin/env python3
"""Greedy / modified-beam search decode + WER/CER cho test set.

Ví dụ:
  python zipformer/decode.py \
    --manifest-dir data/manifests --fbank-dir data/fbank \
    --bpe-model data/lang_bpe_500/bpe.model \
    --exp-dir exp/vi_zipformer \
    --epoch 30 --avg 10 --method modified_beam_search --beam-size 4 \
    --split test --chunk-size 32 --left-context-frames 128
"""
import argparse, logging
from pathlib import Path
import torch, sentencepiece as spm, jiwer

from asr_datamodule import VietnameseAsrDataModule
from model import VietZipformerRNNT
from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.utils import setup_logger


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--fbank-dir", required=True)
    p.add_argument("--bpe-model", required=True)
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--avg", type=int, default=1)
    p.add_argument("--method", default="greedy_search",
                   choices=["greedy_search", "modified_beam_search"])
    p.add_argument("--beam-size", type=int, default=4)
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--max-duration", type=float, default=400)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--left-context-frames", type=int, default=128)
    return p


def greedy_search(model, enc_out, enc_lens, blank_id, max_sym_per_frame=1):
    """Vanilla RNN-T greedy."""
    device = enc_out.device
    N, T, _ = enc_out.shape
    context_size = model.decoder.context_size
    hyps = [[blank_id] * context_size for _ in range(N)]
    dec_in = torch.tensor(hyps, dtype=torch.int64, device=device)
    dec_out = model.decoder(dec_in, need_pad=False)
    for t in range(T):
        am = enc_out[:, t:t+1, :]                    # (N,1,C)
        for _ in range(max_sym_per_frame):
            logits = model.joiner(am, dec_out.unsqueeze(1))  # (N,1,1,V)
            v = logits.squeeze(1).squeeze(1).argmax(-1)      # (N,)
            updated = False
            for n in range(N):
                if t >= enc_lens[n]: continue
                tok = v[n].item()
                if tok != blank_id:
                    hyps[n].append(tok); updated = True
            if not updated: break
            dec_in = torch.tensor([h[-context_size:] for h in hyps],
                                  dtype=torch.int64, device=device)
            dec_out = model.decoder(dec_in, need_pad=False)
    return [h[context_size:] for h in hyps]


def modified_beam_search(model, enc_out, enc_lens, blank_id, beam=4):
    # giản lược: dùng icefall.beam_search nếu sẵn có
    from icefall.decode import modified_beam_search as mbs
    return mbs(model=model, encoder_out=enc_out, encoder_out_lens=enc_lens, beam=beam)


def main():
    args = get_parser().parse_args()
    setup_logger(f"{args.exp_dir}/log/log-decode")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = spm.SentencePieceProcessor(model_file=args.bpe_model)
    model = VietZipformerRNNT(
        vocab_size=sp.get_piece_size(), blank_id=0,
        chunk_size=(args.chunk_size,), left_context_frames=(args.left_context_frames,),
    ).to(device)

    ckpts = [Path(args.exp_dir) / f"epoch-{e}.pt"
             for e in range(args.epoch - args.avg + 1, args.epoch + 1)]
    if args.avg > 1:
        avg = average_checkpoints(ckpts, device=device)
        model.load_state_dict(avg, strict=False)
    else:
        load_checkpoint(ckpts[-1], model=model)
    model.eval()

    dm = VietnameseAsrDataModule(args.manifest_dir, args.fbank_dir,
                                 bpe_model=args.bpe_model,
                                 max_duration=args.max_duration)
    dl = dm.dev_dataloader() if args.split == "dev" else dm.test_dataloader()

    refs, hyps = [], []
    out_f = open(Path(args.exp_dir) / f"hyp-{args.split}-{args.method}.txt", "w", encoding="utf-8")

    with torch.no_grad():
        for batch in dl:
            feats = batch["inputs"].to(device)
            feat_lens = batch["supervisions"]["num_frames"].to(device)
            cuts = batch["supervisions"]["cut"]

            x, x_lens = model.encoder_embed(feats, feat_lens)
            x = x.permute(1, 0, 2)
            enc_out, enc_lens = model.encoder(x, x_lens)
            enc_out = enc_out.permute(1, 0, 2)

            if args.method == "greedy_search":
                hyp_ids = greedy_search(model, enc_out, enc_lens, blank_id=0)
            else:
                hyp_ids = modified_beam_search(model, enc_out, enc_lens, 0, args.beam_size)

            for c, ids in zip(cuts, hyp_ids):
                ref = c.supervisions[0].text
                hyp = sp.decode(ids)
                refs.append(ref); hyps.append(hyp)
                out_f.write(f"{c.id}\tREF\t{ref}\nHYP\t{hyp}\n")

    out_f.close()
    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    logging.info(f"[{args.split}/{args.method}] WER={wer*100:.2f}%  CER={cer*100:.2f}%  "
                 f"N={len(refs)}")
    print(f"WER={wer*100:.2f}% CER={cer*100:.2f}%")


if __name__ == "__main__":
    main()
