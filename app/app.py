"""Streamlit demo: age group WITHOUT vs WITH depth curvature.

    streamlit run app/app.py          (run from the skin_age folder, venv active)
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.curvature import curvature_for_display_masked  # noqa: E402
from models.pipeline import SkinAgePipeline  # noqa: E402

st.set_page_config(page_title="Skin age: with vs without depth curvature", layout="wide")
LABEL = {"young": "Young (10-29)", "middle": "Middle (30-49)", "old": "Old (50+)"}
COLOR = {"young": "#2a9d8f", "middle": "#e9a23b", "old": "#c1553b"}


def group_of(age):
    return None if age < 10 else "young" if age < 30 else "middle" if age < 50 else "old"


@st.cache_resource(show_spinner="Loading the models (first start takes about a minute)...")
def load_pipe():
    return SkinAgePipeline(str(ROOT / "checkpoints" / "final_v1"))


@st.cache_data(show_spinner=False, max_entries=16)
def run(data: bytes, autocrop: bool):
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"error": "This file could not be read as an image."}
    return load_pipe().analyze(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), autocrop=autocrop)


def cmap_rgb(x01, name):
    return (matplotlib.colormaps[name](np.clip(x01, 0, 1))[..., :3] * 255).astype(np.uint8)


def evidence_images(r):
    img = r["img512"]
    wr = img.copy(); wr[r["wrinkle512"] > 0.5] = (255, 20, 20)
    depth = cmap_rgb(r["depth01"], "magma")
    cur = cmap_rgb(curvature_for_display_masked(r["curv"], ~r["keep"], clip_pct=95), "coolwarm")
    cur = np.where(r["keep"][..., None], cur, (r["img256"] // 4).astype(np.uint8))
    return img, wr, depth, cur


def result_card(title, probs, classes, true_group=None):
    k = int(np.argmax(probs)); g = classes[k]
    st.markdown(f"#### {title}")
    st.markdown(f"<div style='font-size:1.9rem;font-weight:700;color:{COLOR[g]}'>{LABEL[g]}</div>", unsafe_allow_html=True)
    st.caption(f"model confidence {probs[k]:.0%}")
    for c, p in zip(classes, probs):
        st.progress(float(p), text=f"{LABEL[c]}  {p:.0%}")
    if true_group:
        if g == true_group:
            st.success("Matches the real age group")
        else:
            st.error(f"Real age group is {LABEL[true_group]}")


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Your photo")
    mode = st.radio("Source", ["Upload a file", "Use the camera"], horizontal=True, label_visibility="collapsed")
    up = st.file_uploader("Face photo", type=["jpg", "jpeg", "png", "webp"]) if mode == "Upload a file" \
        else st.camera_input("Take a photo")
    real_age = st.number_input("Real age (optional, 0 = unknown)", 0, 110, 0)
    autocrop = st.checkbox("Auto-crop to the face", value=True,
                           help="Turn off if your photo is already a tight, front-facing face crop like the FFHQ photos.")
    st.caption("Photos are processed in memory and are not stored by the app.")

tab_predict, tab_about = st.tabs(["Predict", "About / method"])

# --------------------------------------------------------------------------- predict tab
with tab_predict:
    st.title("Skin age group: with and without depth curvature")
    #st.caption("Research prototype from an M.Tech AIML project. Not a medical tool and not for age verification.")
    if up is None:
        st.info("Add a clear, front-facing photo in the sidebar. Whole face visible, even light, no sunglasses.")
    else:
        with st.spinner("Analysing..."):
            r = run(up.getvalue(), autocrop)
        if r["error"]:
            st.error(r["error"])
        else:
            cl = r["classes"]; pw, pf = r["probs_wrinkle"], r["probs_fused"]
            gw, gf = cl[int(np.argmax(pw))], cl[int(np.argmax(pf))]
            tg = group_of(real_age) if real_age else None
            if real_age and tg is None:
                st.warning("Real ages under 10 are outside the three groups, so no right/wrong check is shown.")
            if gw == gf:
                st.success(f"Both methods agree: {LABEL[gf]}")
            else:
                st.warning(f"The methods disagree: {LABEL[gw]} without curvature, {LABEL[gf]} with curvature. "
                           "One photo cannot show which one is right.")
            c1, c2 = st.columns(2, gap="large")
            with c1:
                result_card("Without depth curvature (wrinkle map only)", pw, cl, tg)
            with c2:
                result_card("With depth curvature (wrinkle map + curvature)", pf, cl, tg)

            st.markdown("##### What the model looked at")
            img, wr, depth, cur = evidence_images(r)
            e = st.columns(4)
            e[0].image(img, caption="Face crop used")
            e[1].image(wr, caption="Predicted wrinkles (red)")
            e[2].image(depth, caption="Estimated depth (DPT, frozen)")
            e[3].image(cur, caption="Curvature on the face area (blue concave, red convex)")
            st.caption(f"Face area used: {r['face_frac']:.0%} of the crop | wrinkle pixels inside the face: "
                       f"{r['wrinkle_density']:.1%} | face area kept after removing outline edges: {r['keep_frac']:.0%}")

    with st.expander("How reliable is this?", expanded=up is None):
        st.markdown(
            "- It predicts one of **three groups** (young 10-29, middle 30-49, old 50+), not an exact age.\n"
            "- In 5-fold cross-validation on 1000 photos the accuracy is about **61%** (guessing the biggest group would give 46.5%).\n"
            "- Curvature helped mainly by avoiding big mistakes (young vs old).\n"
            "- The labels are age-group estimates from the FFHQ-Aging dataset, not verified true ages.\n"
            "- Weak spots: glasses (the eye area is skipped), beards, heavy retouching or smoothing, turned heads, strong shadows, "
            "and phone selfies that look different from the training photos.\n"
            "- Percentages are calibrated on held-out data, so a 60% bar means roughly 60 in 100 such cases are right.")

# --------------------------------------------------------------------------- about tab
with tab_about:
    st.header("Method")
    st.markdown(
        "1. **Face finding.** A pretrained face-parsing model locates the face; the photo is cropped like the training photos.\n"
        "2. **Wrinkle map.** A small fusion U-Net (after Moon & Lee 2022) trained on the 1,000 manual masks of FFHQ-Wrinkle.\n"
        "3. **Depth and curvature.** A frozen DPT-Hybrid depth model gives relative depth; mean curvature is computed from it, "
        "limited to the face area, with head-outline edges removed.\n"
        "4. **Two classifiers (MobileNetV3-Small).** One sees only the wrinkle map; the other sees wrinkle map + curvature. "
        "Each is an ensemble of 3 training seeds.\n"
        "5. **Controls.** In the experiments, curvature from a random other photo and a blank channel were tested too, "
        "so the gain is credited to real curvature and not to having an extra input.")
    f = ROOT / "outputs" / "age_cv" / "summary.csv"
    if f.exists():
        st.subheader("Cross-validation results (5 folds, several seeds)")
        d = pd.read_csv(f)
        keep = [c for c in ["config", "acc", "macro_f1", "dF1_vs_wrk", "CI_wrk", "wins_wrk", "dF1_vs_shuf", "CI_shuf",
                            "dF1_vs_zero"] if c in d.columns]
        st.dataframe(d[keep].round(3), hide_index=True)
        st.caption("macro_f1 averages the three groups equally. dF1_vs_* is the gain over wrinkle-only / shuffled-curvature / "
                   "blank-channel controls; CI is a 95% bootstrap interval.")
