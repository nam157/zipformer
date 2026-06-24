#!/usr/bin/env python3
"""Export streaming ZipFormer-RNNT sang ONNX (3 file: encoder/decoder/joiner)
+ dynamic INT8 quantization để chạy CPU."""
import argparse, torch
from pathlib import Path
import sentencepiece as spm
from onnxruntime.quantization import quantize_dynamic, QuantType
from icefall.checkpoint import average_checkpoints, load_checkpoint
from model import VietZipformerRNNT


def export_encoder(model, path, chunk_size, left_ctx):
    feats = torch.randn(1, chunk_size * 2 + 7, 80)
    feat_lens = torch.tensor([feats.size(1)], dtype=torch.int64)

    class Wrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, feats, feat_lens):
            x, xl = self.m.encoder_embed(feats, feat_lens)
            x = x.permute(1, 0, 2)
            o, ol = self.m.encoder(x, xl)
            return o.permute(1, 0, 2), ol

    torch.onnx.export(
        Wrap(model), (feats, feat_lens), path,
        input_names=["feats", "feat_lens"],
        output_names=["enc_out", "enc_out_lens"],
        dynamic_axes={"feats": {0: "N", 1: "T"}, "feat_lens": {0: "N"},
                      "enc_out": {0: "N", 1: "T"}, "enc_out_lens": {0: "N"}},
        opset_version=14,
    )


def export_decoder(model, path):
    ctx = model.decoder.context_size
    y = torch.zeros(1, ctx, dtype=torch.int64)
    torch.onnx.export(
        model.decoder, (y, False), path,
        input_names=["y", "need_pad"], output_names=["dec_out"],
        dynamic_axes={"y": {0: "N"}, "dec_out": {0: "N"}},
        opset_version=14,
    )


def export_joiner(model, path):
    enc_dim = model.joiner.encoder_proj.in_features
    dec_dim = model.joiner.decoder_proj.in_features
    am = torch.randn(1, 1, enc_dim); lm = torch.randn(1, 1, dec_dim)
    torch.onnx.export(
        model.joiner, (am, lm), path,
        input_names=["am", "lm"], output_names=["logits"],
        dynamic_axes={"am": {0: "N"}, "lm": {0: "N"}, "logits": {0: "N"}},
        opset_version=14,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--avg", type=int, default=1)
    ap.add_argument("--bpe-model", required=True)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--left-context-frames", type=int, default=128)
    ap.add_argument("--quantize", type=int, default=1)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.bpe_model)
    model = VietZipformerRNNT(
        vocab_size=sp.get_piece_size(), blank_id=0,
        chunk_size=(args.chunk_size,),
        left_context_frames=(args.left_context_frames,),
    )
    ckpts = [Path(args.exp_dir) / f"epoch-{e}.pt"
             for e in range(args.epoch - args.avg + 1, args.epoch + 1)]
    if args.avg > 1:
        model.load_state_dict(average_checkpoints(ckpts, device="cpu"), strict=False)
    else:
        load_checkpoint(ckpts[-1], model=model)
    model.eval()

    out = Path(args.exp_dir) / "onnx"; out.mkdir(exist_ok=True)
    enc, dec, joi = out/"encoder.onnx", out/"decoder.onnx", out/"joiner.onnx"
    export_encoder(model, enc, args.chunk_size, args.left_context_frames)
    export_decoder(model, dec)
    export_joiner(model, joi)
    print(f"Exported FP32 to {out}")

    if args.quantize:
        for p in [enc, dec, joi]:
            q = p.with_name(p.stem + ".int8.onnx")
            quantize_dynamic(str(p), str(q), weight_type=QuantType.QInt8)
            print(f"INT8 -> {q}")


if __name__ == "__main__":
    main()
