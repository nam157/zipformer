#!/usr/bin/env python3
"""Fine-tune Vietnamese ZipFormer-RNNT streaming.

Ví dụ:
  python zipformer/train.py \
    --manifest-dir data/manifests --fbank-dir data/fbank \
    --bpe-model data/lang_bpe_500/bpe.model \
    --rare-tokens data/lang_bpe_500/rare_tokens.json \
    --exp-dir exp/vi_zipformer \
    --pretrain-ckpt pretrain/zipformer_en_streaming.pt \
    --num-epochs 30 --max-duration 600 --base-lr 0.035 --use-fp16 1
"""
import argparse, logging, math, os
from pathlib import Path
import k2, torch, torch.nn as nn
import sentencepiece as spm
from torch.utils.tensorboard import SummaryWriter

from asr_datamodule import VietnameseAsrDataModule
from model import VietZipformerRNNT

from icefall.checkpoint import save_checkpoint, load_checkpoint
from icefall.utils import AttributeDict, MetricsTracker, setup_logger
from icefall.dist import setup_dist, cleanup_dist
from icefall.lr_scheduler import Eden


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--fbank-dir", required=True)
    p.add_argument("--bpe-model", required=True)
    p.add_argument("--rare-tokens", default=None)
    p.add_argument("--musan-cuts", default=None)
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--pretrain-ckpt", default=None)
    p.add_argument("--init-modules", default="encoder",
                   help="comma list of submodules to load from pretrain")
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--start-epoch", type=int, default=1)
    p.add_argument("--max-duration", type=float, default=600)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--base-lr", type=float, default=0.035)
    p.add_argument("--lr-epochs", type=int, default=6)
    p.add_argument("--lr-batches", type=int, default=7500)
    p.add_argument("--prune-range", type=int, default=5)
    p.add_argument("--am-scale", type=float, default=0.0)
    p.add_argument("--lm-scale", type=float, default=0.0)
    p.add_argument("--simple-loss-scale", type=float, default=0.5)
    p.add_argument("--use-fp16", type=int, default=1)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None,
                   help="ưu tiên hơn --use-fp16 nếu set")
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--grad-checkpoint", type=int, default=0)
    p.add_argument("--tf32", type=int, default=1)
    p.add_argument("--train-manifest", default="cuts_train_fbank.jsonl.gz")
    p.add_argument("--tier-weights", default=None,
                   help="vd: 'clean=2.0,medium=1.0,noisy=0.7'")
    p.add_argument("--source-weights", default=None,
                   help="vd: 'gigaspeech2_vi=0.3,vietmed=3.0'")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-every-n", type=int, default=4000)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p


def load_pretrain(model, ckpt_path, init_modules):
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("model", state)
    keep = {}
    mods = [m.strip() for m in init_modules.split(",") if m.strip()]
    for k, v in sd.items():
        if any(k.startswith(m + ".") for m in mods):
            keep[k] = v
    missing, unexpected = model.load_state_dict(keep, strict=False)
    logging.info(f"Loaded {len(keep)} keys from pretrain. "
                 f"missing={len(missing)} unexpected={len(unexpected)}")


def encode_supervisions(sp: spm.SentencePieceProcessor, supervisions):
    texts = [s.text for s in supervisions]
    ids = [sp.encode(t, out_type=int) for t in texts]
    return k2.RaggedTensor(ids)


def compute_loss(model, sp, batch, args, device):
    feats = batch["inputs"].to(device)                        # (N,T,F)
    feat_lens = batch["supervisions"]["num_frames"].to(device)
    cuts = batch["supervisions"]["cut"]
    sups = [c.supervisions[0] for c in cuts]
    y = encode_supervisions(sp, sups).to(device)

    simple_loss, pruned_loss, _, _ = model(
        feats, feat_lens, y,
        prune_range=args.prune_range,
        am_scale=args.am_scale, lm_scale=args.lm_scale,
    )
    total = args.simple_loss_scale * simple_loss + pruned_loss
    n_frames = feat_lens.sum().item()
    return total, simple_loss.detach(), pruned_loss.detach(), n_frames


def train_one_epoch(model, sp, dl, opt, sched, scaler, args, device, epoch, tb):
    model.train()
    tracker = MetricsTracker()
    accum = max(1, args.grad_accum_steps)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    use_scaler = args.precision == "fp16"
    opt.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dl):
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=(args.precision != "fp32")):
            loss, sl, pl, n_frames = compute_loss(model, sp, batch, args, device)
            loss_scaled = loss / max(1, n_frames) / accum

        if use_scaler:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if (batch_idx + 1) % accum == 0:
            if use_scaler:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step_batch(batch_idx)

        tracker["frames"] += n_frames
        tracker["loss"] += loss.item()
        tracker["simple_loss"] += sl.item()
        tracker["pruned_loss"] += pl.item()
        tracker["batches"] += 1

        if batch_idx % args.log_interval == 0:
            avg = tracker["loss"] / max(1, tracker["frames"])
            lr = opt.param_groups[0]["lr"]
            logging.info(f"epoch {epoch} batch {batch_idx} "
                         f"loss/frame={avg:.4f} lr={lr:.5f}")
            step = epoch * 100000 + batch_idx
            tb.add_scalar("train/loss_per_frame", avg, step)
            tb.add_scalar("train/lr", lr, step)

        if args.save_every_n and batch_idx and batch_idx % args.save_every_n == 0:
            save_checkpoint(
                filename=Path(args.exp_dir) / f"checkpoint-epoch{epoch}-batch{batch_idx}.pt",
                model=model, params=AttributeDict(vars(args)),
                optimizer=opt, scheduler=sched, scaler=scaler, rank=0,
            )


@torch.no_grad()
def validate(model, sp, dl, args, device):
    model.eval()
    total_loss = 0.0; total_frames = 0
    for batch in dl:
        loss, _, _, n = compute_loss(model, sp, batch, args, device)
        total_loss += loss.item(); total_frames += n
    return total_loss / max(1, total_frames)


def run(rank, world_size, args):
    setup_logger(f"{args.exp_dir}/log/log-train")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve precision: --precision ưu tiên, fallback --use-fp16
    if args.precision is None:
        args.precision = "fp16" if args.use_fp16 else "fp32"
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Parse tier / source weights
    def _parse_kv(s):
        out = {}
        for kv in s.split(","):
            k, v = kv.split("="); out[k.strip()] = float(v)
        return out
    tier_w = _parse_kv(args.tier_weights) if args.tier_weights else None
    src_w = _parse_kv(args.source_weights) if args.source_weights else None

    sp = spm.SentencePieceProcessor(model_file=args.bpe_model)
    vocab_size = sp.get_piece_size()
    logging.info(f"vocab_size={vocab_size}")

    model = VietZipformerRNNT(vocab_size=vocab_size, blank_id=0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"#params = {n_params/1e6:.2f}M")

    if args.pretrain_ckpt:
        load_pretrain(model, args.pretrain_ckpt, args.init_modules)

    if args.grad_checkpoint:
        # Bật gradient checkpointing trên encoder để tiết kiệm ~30% VRAM
        from torch.utils.checkpoint import checkpoint_sequential
        if hasattr(model.encoder, "encoders"):
            orig_forward = model.encoder.forward
            def ckpt_forward(x, x_lens, *a, **kw):
                return orig_forward(x, x_lens, *a, **kw)
            # icefall Zipformer2 hỗ trợ flag `use_grad_checkpoint`
            for m in model.encoder.modules():
                if hasattr(m, "use_grad_checkpoint"):
                    m.use_grad_checkpoint = True

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    dm = VietnameseAsrDataModule(
        manifest_dir=args.manifest_dir, fbank_dir=args.fbank_dir,
        musan_cuts=args.musan_cuts, rare_tokens_json=args.rare_tokens,
        bpe_model=args.bpe_model,
        max_duration=args.max_duration, num_workers=args.num_workers,
    )
    train_dl = dm.train_dataloader(manifest_name=args.train_manifest,
                                    tier_weights=tier_w, source_weights=src_w)
    dev_dl = dm.dev_dataloader()

    opt = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-2)
    sched = Eden(opt, args.lr_batches, args.lr_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "fp16"))

    tb = SummaryWriter(log_dir=f"{args.exp_dir}/tb")

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        sched.step_epoch(epoch - 1)
        train_one_epoch(model, sp, train_dl, opt, sched, scaler, args, device, epoch, tb)
        val_loss = validate(model, sp, dev_dl, args, device)
        logging.info(f"epoch {epoch} dev loss/frame = {val_loss:.4f}")
        tb.add_scalar("dev/loss_per_frame", val_loss, epoch)
        save_checkpoint(
            filename=Path(args.exp_dir) / f"epoch-{epoch}.pt",
            model=model.module if hasattr(model, "module") else model,
            params=AttributeDict(vars(args)),
            optimizer=opt, scheduler=sched, scaler=scaler, rank=0,
        )

    if world_size > 1:
        cleanup_dist()


def main():
    args = get_parser().parse_args()
    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    if args.world_size > 1:
        torch.multiprocessing.spawn(run, args=(args.world_size, args),
                                    nprocs=args.world_size, join=True)
    else:
        run(0, 1, args)


if __name__ == "__main__":
    main()
