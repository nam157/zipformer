#!/usr/bin/env python3
"""Wrapper gọi ZipFormer streaming + RNN-T loss của k2.
Dùng các module có sẵn trong icefall (đã clone vào PYTHONPATH)."""
import torch, torch.nn as nn, k2
from icefall.utils import make_pad_mask

from icefall.models.zipformer.zipformer import Zipformer2
from icefall.models.zipformer.subsampling import Conv2dSubsampling
from icefall.models.zipformer.decoder import Decoder
from icefall.models.zipformer.joiner import Joiner


class VietZipformerRNNT(nn.Module):
    def __init__(self, vocab_size: int, blank_id: int = 0,
                 feature_dim: int = 80,
                 encoder_dim=(192, 256, 384, 512, 384, 256),
                 num_encoder_layers=(2, 2, 3, 4, 3, 2),
                 feedforward_dim=(512, 768, 1024, 1536, 1024, 768),
                 encoder_unmasked_dim=(192, 192, 256, 256, 256, 192),
                 decoder_dim: int = 512, joiner_dim: int = 512,
                 causal: bool = True,
                 chunk_size=(16, 32, 64), left_context_frames=(64, 128, 256)):
        super().__init__()
        self.encoder_embed = Conv2dSubsampling(
            in_channels=feature_dim, out_channels=encoder_dim[0],
            dropout=0.1,
        )
        self.encoder = Zipformer2(
            output_downsampling_factor=2,
            downsampling_factor=(1, 2, 4, 8, 4, 2),
            num_encoder_layers=num_encoder_layers,
            encoder_dim=encoder_dim,
            encoder_unmasked_dim=encoder_unmasked_dim,
            feedforward_dim=feedforward_dim,
            num_heads=(4, 4, 4, 8, 4, 4),
            causal=causal,
            chunk_size=list(chunk_size),
            left_context_frames=list(left_context_frames),
        )
        self.decoder = Decoder(vocab_size=vocab_size, decoder_dim=decoder_dim,
                               blank_id=blank_id, context_size=2)
        self.joiner = Joiner(encoder_dim=max(encoder_dim),
                             decoder_dim=decoder_dim,
                             joiner_dim=joiner_dim,
                             vocab_size=vocab_size)
        self.blank_id = blank_id
        self.vocab_size = vocab_size

    def forward(self, feats, feat_lens, y: k2.RaggedTensor,
                prune_range: int = 5, am_scale: float = 0.0, lm_scale: float = 0.0):
        # encoder
        x, x_lens = self.encoder_embed(feats, feat_lens)
        x = x.permute(1, 0, 2)  # T,N,C
        enc_out, enc_out_lens = self.encoder(x, x_lens)
        enc_out = enc_out.permute(1, 0, 2)  # N,T,C

        # decoder (prepend blank)
        sos_y = k2.ragged.add_suffix(y, self.blank_id)  # add blank at end as sos shift
        y_padded = sos_y.pad(mode="constant", padding_value=self.blank_id)
        dec_out = self.decoder(y_padded)

        # pruned RNN-T (k2)
        y_lens = y.compute_lengths()
        boundary = torch.zeros(feats.size(0), 4, dtype=torch.int64, device=feats.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = enc_out_lens

        simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
            lm=dec_out, am=enc_out,
            symbols=y_padded, termination_symbol=self.blank_id,
            lm_only_scale=lm_scale, am_only_scale=am_scale,
            boundary=boundary, reduction="sum", return_grad=True,
        )
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad, py_grad=py_grad, boundary=boundary, s_range=prune_range,
        )
        am_pruned, lm_pruned = k2.do_rnnt_pruning(am=enc_out, lm=dec_out, ranges=ranges)
        logits = self.joiner(am_pruned, lm_pruned)
        pruned_loss = k2.rnnt_loss_pruned(
            logits=logits, symbols=y_padded, ranges=ranges,
            termination_symbol=self.blank_id, boundary=boundary, reduction="sum",
        )
        return simple_loss, pruned_loss, enc_out, enc_out_lens
