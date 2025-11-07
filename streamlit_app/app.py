import io
import os
import gc
import time
import base64
import requests
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# ------------------ Safety & UX ------------------
st.set_page_config(page_title="RealEyes", layout="centered")
st.title("RealEyes — Streamlit Demo")

st.write(
    "Upload an image and get a segmentation mask. "
    )

# ------------------ Config ------------------
MAX_UPLOAD_MB     = 5            # hard cap on input file
MAX_RESOLUTION    = 512          # long-side cap before sending to API
JPEG_QUALITY      = 90           # final client->API JPEG quality
REQUEST_TIMEOUT_S = 30

def default_api_base() -> str:
    env_api = os.getenv("API_URL")
    if env_api:
        return env_api.rstrip("/")
    if os.getenv("KUBERNETES_SERVICE_HOST"):  # set inside pods
        return "http://deepfake-api-svc.deepfake.svc.cluster.local:8000"
    return "http://deepfake.local/api"

API_BASE = st.text_input("API base URL", default_api_base()).rstrip("/")

# Use session_state to hold ONLY a small, compressed JPEG for inference
if "upload_jpeg" not in st.session_state:
    st.session_state.upload_jpeg = None
if "preview_shape" not in st.session_state:
    st.session_state.preview_shape = None  # (w, h) after downscale

# ------------------ File uploader ------------------
file = st.file_uploader("Upload an image (≤ 5 MB)", type=["jpg", "jpeg", "png", "webp", "bmp"])

def prepare_upload_bytes(_file) -> tuple[bytes, tuple[int, int]]:
    """
    Load image safely, downscale to MAX_RESOLUTION, compress to JPEG,
    return (jpeg_bytes, (w,h) of downscaled image).
    """
    # size check (streamlit provides .size)
    size_mb = (_file.size or 0) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(f"File too large: {size_mb:.2f} MB (max {MAX_UPLOAD_MB} MB).")

    try:
        img = Image.open(_file).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Invalid or unsupported image file.")

    # early downscale to cap memory footprint
    w, h = img.size
    if max(w, h) > MAX_RESOLUTION:
        img.thumbnail((MAX_RESOLUTION, MAX_RESOLUTION))  # in-place, preserves aspect
        w, h = img.size

    # compress to JPEG (keeps memory tiny vs raw arrays)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    jpeg_bytes = buf.getvalue()

    # free heavy objects ASAP
    buf.close()
    del buf, img
    gc.collect()

    return jpeg_bytes, (w, h)

if file is not None:
    try:
        st.session_state.upload_jpeg, st.session_state.preview_shape = prepare_upload_bytes(file)
        # Show a lightweight preview from the compressed bytes
        st.image(st.session_state.upload_jpeg, caption=f"Preview (max side {MAX_RESOLUTION}px)", use_column_width=True)
    except Exception as e:
        st.session_state.upload_jpeg = None
        st.session_state.preview_shape = None
        st.error(str(e))

# ------------------ Health / Metrics ------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("Health"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=5)
            r.raise_for_status()
            st.code(r.json())
        except Exception as e:
            st.error(f"Health request failed: {e}")

with col2:
    if st.button("Metrics (head)"):
        try:
            r = requests.get(f"{API_BASE}/metrics", timeout=5)
            r.raise_for_status()
            st.text("\n".join(r.text.splitlines()[:15]))
        except Exception as e:
            st.error(f"Metrics request failed: {e}")

st.markdown("---")

# ------------------ Predict (OOM-safe) ------------------
predict_btn = st.button("Predict", disabled=(st.session_state.upload_jpeg is None))
if predict_btn and st.session_state.upload_jpeg is not None:
    t0 = time.perf_counter()
    try:
        # stream the small JPEG to API
        with io.BytesIO(st.session_state.upload_jpeg) as buf:
            files = {"file": ("image.jpg", buf, "image/jpeg")}
            resp = requests.post(f"{API_BASE}/predict", files=files, timeout=REQUEST_TIMEOUT_S)

        if resp.status_code != 200:
            short = resp.text[:300].replace("\n", " ")
            st.error(f" {resp.status_code}: {short} …")
        else:
            js = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000.0
            st.success(f" Inference OK — {elapsed:.0f} ms")

            # Decode masks as lightweight PILs (avoid giant arrays early)
            soft_png = base64.b64decode(js["mask_base64"])
            soft_img = Image.open(io.BytesIO(soft_png)).convert("L")

            bin_src = js.get("mask_bin_base64", js["mask_base64"])
            bin_png = base64.b64decode(bin_src)
            bin_img = Image.open(io.BytesIO(bin_png)).convert("L")

            # For overlay, reconstruct a preview RGB from the compressed upload
            base_img = Image.open(io.BytesIO(st.session_state.upload_jpeg)).convert("RGB")
            # Resize masks to preview size (they may already be 256x256 from model)
            soft_img = soft_img.resize(base_img.size)
            bin_img  = bin_img.resize(base_img.size)

            # Overlay in NumPy (keep lifetime short)
            img_np  = np.asarray(base_img, dtype=np.float32)
            mask_np = np.asarray(soft_img, dtype=np.float32) / 255.0

            alpha = 0.40
            red   = np.array([255.0, 0.0, 0.0], dtype=np.float32)
            overlay = img_np * (1.0 - alpha * mask_np[..., None]) + red * (alpha * mask_np[..., None])
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            overlay_img = Image.fromarray(overlay)

            # Show results
            c1, c2 = st.columns(2)
            with c1:
                st.image(base_img,  caption="Original (downscaled)", use_column_width=True)
                st.image(overlay_img, caption="Overlay (Predicted Mask)", use_column_width=True)
            with c2:
                st.image(soft_img, caption="Soft mask (0–255)", use_column_width=True)
                st.image(bin_img,  caption="Binary mask", use_column_width=True)

            with st.expander("Raw JSON Response"):
                st.json(js)

            # Aggressive cleanup of large objects
            del img_np, mask_np, overlay
            del base_img, soft_img, bin_img, overlay_img
            gc.collect()

    except Exception as e:
        st.error(f"Predict failed: {e}")
    finally:
        # No need to keep large objects around after one prediction.
        # Keep only the tiny compressed upload for quick re-tries.
        gc.collect()
