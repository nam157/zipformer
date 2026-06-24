# Mix 6000h — phân tích & chiến lược sampling

| Source           | Giờ ước tính | Tier   | Note                                   |
|------------------|--------------|--------|----------------------------------------|
| VLSP2020         | ~150h        | medium | broadcast + read, label gốc OK         |
| VLSP2021         | ~200h        | medium | mở rộng VLSP2020                       |
| VLSP2023         | ~250h        | medium | có phần spontaneous                    |
| FPT FOSD         | ~25h         | clean  | đọc rõ, studio                         |
| VIET_BUD500      | ~500h        | noisy  | YouTube auto-sub                       |
| VietSpeech       | ~100h        | medium | NhutP/VSV-1100                         |
| FLEURS-VI        | ~12h         | clean  | parallel multilingual                  |
| VietMed_Labeled  | ~16h         | clean  | medical domain (quan trọng)            |
| Sub-GigaSpeech2  | ~3000h       | noisy  | YouTube, **dominant** — phải cap       |
| ViVoice          | ~1000h       | clean  | TTS-grade, có thể overrepresent đọc    |
| Sub-PhoAudioBook | ~200h        | clean  | audiobook, đọc đều giọng               |
| **TỔNG**         | **~5450h**   |        |                                        |

## Vấn đề distribution
- GigaSpeech2 (3000h) + ViVoice (1000h) **chiếm 73%** → model sẽ bias mạnh về YouTube ASR-style và TTS-style.
- Spontaneous/conversational (VLSP2023, BUD500 phần thoại) chỉ ~500h.
- Medical (VietMed) chỉ 16h nhưng giá trị cao → **upsample mạnh**.

## Chiến lược tier-weights (sampling factor)

### Stage A — clean-only (~1250h hiệu dụng)
- Dùng: FOSD, FLEURS, VietMed (×3), ViVoice (×0.5), PhoAudioBook
- ViVoice cap xuống ×0.5 vì tránh model học bias TTS prosody
- VietMed ×3 vì domain hiếm

```
--train-manifest cuts_train_clean_fbank.jsonl.gz
--tier-weights "clean=1.0"
--source-weights "vietmed=3.0,vivoice=0.5,fleurs_vi=2.0"
```

### Stage B — full mix với cap mạnh
- GigaSpeech2: ×0.3 (giảm dominance từ 3000→900h hiệu dụng)
- BUD500: ×0.5
- ViVoice: ×0.6
- VLSP2020/21/23: ×1.5 (medium nhưng giá trị cao)
- FOSD/FLEURS/VietMed/PhoAudioBook: ×2.5
- VietSpeech: ×1.5

```
--train-manifest cuts_train_full_fbank.jsonl.gz
--source-weights "gigaspeech2_vi=0.3,bud500=0.5,vivoice=0.6,vlsp2020=1.5,vlsp2021=1.5,vlsp2023=1.5,fosd=2.5,fleurs_vi=2.5,vietmed=2.5,phoaudiobook=2.5,vietspeech=1.5"
```

→ Hiệu dụng ~3200h, phân bố cân bằng hơn:
- clean ~30%, medium ~35%, noisy ~35%
- domain spread: read 35%, conversational 25%, broadcast 20%, youtube 20%

## Whisper refine — áp ngưỡng theo tier
- clean: skip (label đã tốt)
- medium: max_wer=0.25
- noisy: max_wer=0.5 (BUD500, GigaSpeech2 lỗi nhiều)

## Vocab size đề xuất
- 6000h → BPE **2000** (vocab 500 underfit với corpus lớn này)
- Hoặc unigram 4000 nếu muốn tách phụ âm đầu/vần tiếng Việt tốt hơn
