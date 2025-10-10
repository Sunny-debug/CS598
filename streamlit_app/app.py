import io
import time
import base64
import os
import requests
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("Deepfake Detector — Streamlit Demo")

st.write(
    "Upload an image and get a segmentation mask showing edited regions. "
    "This connects to your FastAPI microservice running on Docker, Minikube, or localhost."
)

# ---------- Smart default for API base ----------
# Priority:
# 1) API_URL env if provided
# 2) If running inside k8s -> internal Service DNS
# 3) Otherwise -> public ingress host (dev)
def default_api_base() -> str:
    env_api = os.getenv("API_URL")
    if env_api:
        return env_api
    if os.getenv("KUBERNETES_SERVICE_HOST"):  # set inside pods
        return "http://deepfake-api:8000"
    # local dev via ingress (ensure /etc/hosts + minikube tunnel)
    return "http://deepfake.local/api"

api_url = st.text_input("API base URL", default_api_base())

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])

# ---------- Health / Metrics ----------
col1, col2 = st.columns(2)
with col1:
    if st.button("Health"):
        try:
            r = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
            r.raise_for_status()
            st.code(r.json())
        except Exception as e:
            st.error(f"Health request failed: {e}")

with col2:
    if st.button("Metrics (head)"):
        try:
            r = requests.get(f"{api_url.rstrip('/')}/metrics", timeout=5)
            r.raise_for_status()
            st.text("\n".join(r.text.splitlines()[:15]))
        except Exception as e:
            st.error(f"Metrics request failed: {e}")

st.markdown("---")

# ---------- Prediction workflow ----------
if st.button("Predict") and file:
    try:
        img = Image.open(file).convert("RGB")
        t0 = time.time()
        r = requests.post(
            f"{api_url.rstrip('/')}/predict",
            files={"file": (file.name, file.getvalue(), file.type or "image/jpeg")},
            timeout=60,
        )
        dt = (time.time() - t0) * 1000

        if not r.ok:
            short = r.text[:300].replace("\n", " ")
            st.error(f"❌ {r.status_code}: {short} …")
        else:
            js = r.json()
            st.success(f"✅ Inference OK — {dt:.0f} ms")

            # Decode mask (soft)
            soft_png = base64.b64decode(js["mask_base64"])
            mask = Image.open(io.BytesIO(soft_png)).convert("L").resize(img.size)

            # Optional: decode binary mask for separate preview
            bin_png = base64.b64decode(js.get("mask_bin_base64", js["mask_base64"]))
            mask_bin = Image.open(io.BytesIO(bin_png)).convert("L").resize(img.size)

            # Build red overlay on original
            img_np  = np.asarray(img, dtype=np.float32)
            mask_np = np.asarray(mask, dtype=np.float32) / 255.0
            alpha   = 0.40
            red     = np.array([255, 0, 0], dtype=np.float32)

            m3 = mask_np[..., None]
            overlay = img_np * (1.0 - alpha * m3) + red * (alpha * m3)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            overlay_img = Image.fromarray(overlay)

            c1, c2 = st.columns(2)
            with c1:
                st.image(img, caption="Original", use_column_width=True)
                st.image(overlay_img, caption="Overlay (Predicted Mask)", use_column_width=True)
            with c2:
                st.image(mask, caption="Soft mask (0–255)", use_column_width=True)
                st.image(mask_bin, caption="Binary mask", use_column_width=True)

            with st.expander("Raw JSON Response"):
                st.json(js)

    except Exception as e:
        st.error(f"Predict failed: {e}")