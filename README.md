<<<<<<< HEAD
# Curvature-aware skin aging analysis (M.Tech project)

Fusion U-Net wrinkle segmentation + frozen MiDaS-small curvature branch -> age estimate.
Streamlit app compares results **with** and **without** the curvature branch.

## Folder layout
```
data/         dataset loaders / splits (later)
models/       curvature.py (done), fusion_unet.py, heads.py (later)
train/        training scripts (later)
eval/         DSC, JSC, MHD, ARI, VI + ablation (later)
app/          streamlit_app.py (later)
scripts/      01_midas_curvature_demo.py  <- you are here
samples/      put 5-10 face images here for the demo
checkpoints/  trained weights (git-ignored)
```

## Step 0 - local setup (Windows, RTX 2050)
```powershell
cd skin_age
python -m venv .venv
.venv\Scripts\activate
# install CUDA build of PyTorch from https://pytorch.org (Get Started page gives the exact command)
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expect `True NVIDIA GeForce RTX 2050`. If it prints `False`, the CPU-only torch got installed.

## Step 1 - check the curvature idea
1. Offline math check (no GPU/internet needed): `python scripts/test_curvature.py`
2. Put 5-10 face photos in `samples/` (some young, some old, a few with strong shading).
3. `python scripts/01_midas_curvature_demo.py --images samples --out outputs/demo`
4. Open `outputs/demo/*.png` (input | depth | curvature).

**What to look for:** does the curvature map follow real face shape (nose, cheeks, jaw) rather than
just noise or wrinkle texture? Try `--sigma 1`, `2`, `4`. Bring a few of the output images
back so we can decide the smoothing and how to feed curvature into the fusion layer.

## Kaggle
Push this repo to GitHub, then in a Kaggle notebook (Internet ON, GPU ON):
```
!git clone https://github.com/<you>/skin_age.git
%cd skin_age
!pip install -q timm
!python scripts/01_midas_curvature_demo.py --images /kaggle/input/<your-dataset> --out /kaggle/working/demo
```
=======
# skin-age-detetction
>>>>>>> b9c5ab5c4919adf4e83f14d4c5d5fff817d4b165
