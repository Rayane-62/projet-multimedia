# MPEG-4-lite encoder (Python)

Simplified MPEG-4-like video codec — M1 IL G3 — USTHB 2025/2026

**BENLALAM Mohamed Rayane** — 222231363816  
**AKKOUCHE Mehdi** — 222231370206

---

## Pipeline

| Étape | Technique |
|-------|-----------|
| 1. Pré-traitement | BGR → YCbCr (BT.601) + sous-échantillonnage 4:2:0 |
| 2. I-frames | DCT 8×8 (`cv2.dct`) + tables de quantification JPEG |
| 3. P-frames | Three-Step Search sur macroblocs 16×16 + DCT du résidu |
| 4. Entropie | Bitstream struct-packed → `bz2` niveau 9 |
| 5. Evaluation | PSNR, ratio de compression, figure matplotlib |

## Fichiers

```
mpeg_codec.py       - fonctions codec (encode, decode, helpers)
gen_test_frames.py  - générer un clip synthétique de test
viz.py              - figure pipeline + courbes d'analyse
run.py              - CLI (encode / decode / viz / sweep)
report.pdf          - rapport du projet
```

## Installation

```bash
pip install numpy opencv-python matplotlib
```

## Utilisation

```bash
# Générer des frames de test
python gen_test_frames.py -o sample_frames -n 12

# Encoder
python run.py encode sample_frames -o video.bin --gop 8 --q 50

# Décoder + PSNR
python run.py decode video.bin -o decoded --ref sample_frames

# Figure pipeline
python run.py viz sample_frames video.bin -o pipeline.png

# Courbes d'analyse
python run.py sweep sample_frames -o experiments.png
```

## Résultats (Q=50, GOP=8, 12 frames 128×128)

- Ratio de compression : **~134×**
- PSNR moyen : **~31.3 dB**
- I-frames : 2 | P-frames : 10
