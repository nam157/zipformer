#!/usr/bin/env python3
"""DataModule: load fbank cuts, apply SpecAugment + noise + speed perturb + OOV oversample."""
import json
from pathlib import Path
from functools import lru_cache
import torch
from lhotse import CutSet, load_manifest_lazy
from lhotse.dataset import (
    K2SpeechRecognitionDataset, DynamicBucketingSampler,
    SpecAugment, CutMix, PerturbSpeed,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader


class VietnameseAsrDataModule:
    def __init__(self, manifest_dir: str, fbank_dir: str,
                 musan_cuts: str = None, rare_tokens_json: str = None,
                 bpe_model: str = None,
                 max_duration: float = 600, num_workers: int = 4,
                 oversample_factor: int = 4):
        self.manifest_dir = Path(manifest_dir)
        self.fbank_dir = Path(fbank_dir)
        self.musan = musan_cuts
        self.max_duration = max_duration
        self.num_workers = num_workers
        self.oversample_factor = oversample_factor

        self.rare = set()
        if rare_tokens_json and Path(rare_tokens_json).exists():
            self.rare = set(json.load(open(rare_tokens_json))["rare_tokens"])

        self._sp = None
        if bpe_model:
            import sentencepiece as spm
            self._sp = spm.SentencePieceProcessor(model_file=bpe_model)

    def _maybe_oversample(self, cuts: CutSet) -> CutSet:
        if not self.rare or self._sp is None:
            return cuts
        rare = self.rare; sp = self._sp; k = self.oversample_factor
        def gen():
            for c in cuts:
                yield c
                text = c.supervisions[0].text if c.supervisions else ""
                toks = set(sp.encode(text, out_type=str))
                if toks & rare:
                    for i in range(k - 1):
                        yield c.with_id(f"{c.id}_dup{i}")
        return CutSet.from_cuts(list(gen()))

    def train_dataloader(self, manifest_name: str = "cuts_train_fbank.jsonl.gz",
                          tier_weights: dict = None, source_weights: dict = None):
        """tier_weights: {'clean':2.0,'medium':1.0,'noisy':0.7}
           source_weights: {'gigaspeech2_vi':0.3,'vietmed':3.0,...}
        Source weight ưu tiên hơn tier khi cả 2 cùng có cho 1 cut."""
        cuts = load_manifest_lazy(self.manifest_dir / manifest_name)
        if tier_weights or source_weights:
            def gen():
                for c in cuts:
                    meta = c.supervisions[0].custom or {}
                    src, tier = meta.get("source"), meta.get("tier", "medium")
                    w = None
                    if source_weights and src in source_weights:
                        w = source_weights[src]
                    elif tier_weights:
                        w = tier_weights.get(tier, 1.0)
                    if w is None: w = 1.0
                    n_int = int(w)
                    frac = w - n_int
                    n = n_int + (1 if (hash(c.id) % 1000) / 1000.0 < frac else 0)
                    if n <= 0:
                        # subsample: bỏ qua với xác suất (1-w)
                        if (hash(c.id) % 1000) / 1000.0 < w: yield c
                        continue
                    for i in range(n):
                        yield c if i == 0 else c.with_id(f"{c.id}_w{i}")
            cuts = CutSet.from_cuts(list(gen()))
        cuts = self._maybe_oversample(cuts)

        transforms = []
        if self.musan and Path(self.musan).exists():
            musan = load_manifest_lazy(self.musan)
            transforms.append(CutMix(cuts=musan, snr=(10, 20), p=0.5))
        transforms.append(PerturbSpeed(factors=[0.9, 1.1], p=0.5))

        input_transforms = [
            SpecAugment(num_feature_masks=2, features_mask_size=27,
                        num_frame_masks=2, frames_mask_size=100, p=0.9)
        ]

        ds = K2SpeechRecognitionDataset(
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=True,
        )
        sampler = DynamicBucketingSampler(
            cuts, max_duration=self.max_duration, shuffle=True,
            num_buckets=30, drop_last=True,
        )
        return DataLoader(ds, sampler=sampler, batch_size=None,
                          num_workers=self.num_workers, persistent_workers=True)

    def _eval_loader(self, name: str):
        cuts = load_manifest_lazy(self.manifest_dir / f"cuts_{name}_fbank.jsonl.gz")
        ds = K2SpeechRecognitionDataset(return_cuts=True)
        sampler = DynamicBucketingSampler(cuts, max_duration=self.max_duration,
                                          shuffle=False, drop_last=False)
        return DataLoader(ds, sampler=sampler, batch_size=None,
                          num_workers=self.num_workers)

    def dev_dataloader(self):  return self._eval_loader("dev")
    def test_dataloader(self): return self._eval_loader("test")
