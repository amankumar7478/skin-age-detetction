# Curvature-Aware Facial Skin-Age Group Classification

A controlled study of whether **surface curvature estimated from a single photo** (frozen monocular depth network) adds age information to a **wrinkle map**. Built as an M.Tech AIML project at NSUT, after the fusion U-Net of Moon & Lee (IEEE Access, 2022).

> Research prototype. It estimates an *apparent age group* (10–29, 30–49, 50+) for study purposes only. It is not a medical tool and must not be used for age verification or any decision about people.

## What it does

```
face photo ──► face region (SegFormer parsing, crop)
   ├─► Fusion U-Net ───────────────► wrinkle map ─────────┐
   └─► frozen DPT-Hybrid depth ─► mean curvature C = K ⊙ H ┴─► MobileNetV3-Small ─► age group
                                   (face mask, outline removed)      (with / without curvature)
```

- **Wrinkle branch:** 5-level fusion U-Net (1.96 M parameters), BCE + soft Dice, trained on FFHQ-Wrinkle manual masks. Five-fold Dice 0.586 ± 0.012.
- **Shape branch:** DPT-Hybrid (MiDaS) depth → Gaussian smoothing → mean curvature → face mask and head-outline suppression.
- **Classifier:** MobileNetV3-Small (ImageNet weights), 3 classes. Probabilities are calibrated with temperature scaling.
- **Controls:** curvature of a random *other* face (shuffled) and an all-zero extra channel, to show that any gain is not just "one more input channel".

## Results

740 FFHQ photos, 5 folds × 5 seeds, macro-F1 (majority-class accuracy = 0.465):

| Variant | Macro-F1 |
|---|---|
| Curvature only | 0.492 |
| Wrinkle only (baseline) | 0.536 |
| Wrinkle + blank channel (control) | 0.547 |
| Wrinkle + shuffled curvature (control) | 0.576 |
| Wrinkle + curvature σ=2 | 0.583 |
| **Wrinkle + curvature σ=4 (proposed)** | **0.612** |

Proposed vs wrinkle only: +0.076 (95% CI +0.049 to +0.101, 5/5 seeds). Proposed vs shuffled control: +0.036 (95% CI +0.007 to +0.066, 5/5 seeds).

**Honest summary:** the gain is small but consistent. Curvature mainly removes severe young↔old mistakes (young predicted old: 80 → 17). It does **not** reduce old faces predicted as middle-aged (~35% in all variants). Better wrinkle masks barely help the wrinkle-only model.

## Limitations

- Labels are crowd-sourced *apparent* age groups (FFHQ-Aging), not verified ages.
- Only 740 photos from one web-collected dataset (148 in the old class); no external test set.
- No plain-photo (RGB) baseline yet, so we cannot say how much age signal wrinkles + curvature capture compared with the photo itself.
- Depth is a frozen generic network's guess, not a measurement.
- Glasses, beards, heavy retouching, strong head pose and several faces in one image give unreliable results.

## Repository layout

```
app/app.py                    Streamlit demo (with vs without curvature, calibrated confidence)
models/                       curvature.py, fusion_unet.py, face_region.py, pipeline.py
data/dataset.py               dataset loaders
train/train_seg.py            wrinkle U-Net (one held-out fold)
train/run_age_cv.py           age classifiers + ablation (5 folds × seeds)
train/train_final.py          final models for the app (all data, 3-seed ensemble)
scripts/                      01–12: data checks, folds, curvature cache, folder prediction, calibration
requirements.txt
```

Datasets and trained weights are **not** included (see Data).

## Setup (Windows / Linux)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
# install the CUDA build of PyTorch first from https://pytorch.org, then:
pip install -r requirements.txt
pip install transformers      # SegFormer face parsing
python -c "import torch; print(torch.cuda.is_available())"
```

Tested on Python 3.13, PyTorch (CUDA 12.6), RTX 2050 (4 GB).

## Data

Download these yourself and respect their licences:

1. **FFHQ-Wrinkle** (1,000 manual masks + parsing labels): <https://arxiv.org/abs/2408.10060>
2. **FFHQ photographs** (512×512 copy is enough): <https://github.com/NVlabs/ffhq-dataset> (CC BY-NC-SA 4.0 for the dataset)
3. **FFHQ-Aging labels**: <https://github.com/royorel/FFHQ-Aging-Dataset>

Expected folders: `manual_wrinkle_masks/`, `face parsed label/`, and the labels CSV `ffhq_aging_labels.csv` in the project root.

## Reproduce the experiments

```bash
python scripts/02_inspect_data.py                 # data summary
python scripts/03_make_id_list.py                 # ids.txt of annotated photos
python scripts/04_extract_subset.py --src <path to FFHQ 512 images>
python scripts/05_check_alignment.py              # masks line up with photos
python scripts/06_make_folds.py                   # 5 stratified folds
python scripts/07_precompute_curvature.py         # cache depth + curvature (GPU free)

for f in 0 1 2 3 4; do python train/train_seg.py --fold $f --epochs 40; done   # out-of-fold wrinkle maps
python train/run_age_cv.py --seeds 5              # ablation, all variants
python train/run_age_cv.py --wrinkle-source human --seeds 3 --out outputs/age_cv_human   # human-mask check
```

Results (accuracy, macro-F1, recalls, paired differences, bootstrap intervals) are written to `outputs/age_cv/summary.csv`.

## Train the app models and run the demo

```bash
python train/train_final.py                 # writes checkpoints/final_v1/
python scripts/12_calibrate_probs.py        # temperature scaling -> calibration.json
python -m streamlit run app/app.py
```

Predict a folder of photos without the app:

```bash
python scripts/09_predict_folder.py --folder my_photos --out outputs/my_results
python scripts/11_compare_truth.py --pred outputs/my_results/results.csv --truth truth.csv
```

## Paper

The accompanying write-up is in `paper/` (LaTeX, IEEEtran). Add: `[PAPER LINK]`.

## Citation

```bibtex
@misc{YOURNAME2026curvature,
  author = {YOUR FULL NAME},
  title  = {Curvature-Aware Facial Skin-Age Group Classification},
  year   = {2026},
  note   = {M.Tech project, Netaji Subhas University of Technology},
  url    = {https://github.com/YOUR-USERNAME/YOUR-REPO}
}
```

## Acknowledgements and references

- C.-I. Moon and O. Lee, "Skin microstructure segmentation and aging classification using CNN-based models," *IEEE Access* 10, 2022. <https://doi.org/10.1109/ACCESS.2021.3140031>
- J. Moon, H. Chung, I. Jang, FFHQ-Wrinkle, ICPR 2024. <https://arxiv.org/abs/2408.10060>
- R. Or-El et al., FFHQ-Aging, ECCV 2020. <https://arxiv.org/abs/2003.09764>
- T. Karras et al., FFHQ / StyleGAN, CVPR 2019. <https://arxiv.org/abs/1812.04948>
- R. Ranftl et al., MiDaS (TPAMI 2022) <https://arxiv.org/abs/1907.01341> and DPT (ICCV 2021) <https://arxiv.org/abs/2103.13413>
- A. Howard et al., MobileNetV3, ICCV 2019. <https://arxiv.org/abs/1905.02244>
- E. Xie et al., SegFormer, NeurIPS 2021. <https://arxiv.org/abs/2105.15203>; face-parsing model: <https://huggingface.co/jonathandinu/face-parsing>
- C. Guo et al., temperature scaling, ICML 2017. <https://arxiv.org/abs/1706.04599>

## Licence

`[CHOOSE: e.g. MIT for the code]`. The datasets and pretrained models keep their own licences; FFHQ-derived data is non-commercial.
