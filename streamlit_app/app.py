import time
import requests
import streamlit as st
import os

st.set_page_config(page_title="Deepfake Detector Demo", layout="centered")
st.title("Deepfake Detector — Streamlit Demo")

# Pick API base URL from environment variable if set, otherwise fallback to localhost:30080
api_url_default = os.getenv(
    "API_BASE",
    "http://{}:{}".format(
        st.session_state.get("node_ip", "127.0.0.1"),
        st.session_state.get("node_port", "30080"),
    ),
)

api_url = st.text_input("API base URL", api_url_default)

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])

col1, col2 = st.columns(2)
with col1:
    if st.button("Health"):
        try:
            r = requests.get(f"{api_url}/healthz", timeout=5)
            st.code(r.json())
        except Exception as e:
            st.error(f"Health request failed: {e}")

with col2:
    if st.button("Metrics (head)"):
        try:
            r = requests.get(f"{api_url}/metrics", timeout=5)
            st.text("\n".join(r.text.splitlines()[:15]))
        except Exception as e:
            st.error(f"Metrics request failed: {e}")

st.markdown("---")

if st.button("Predict") and file:
    try:
        t0 = time.time()
        r = requests.post(
            f"{api_url}/predict",
            files={"file": (file.name, file.getvalue(), file.type or "image/jpeg")},
            timeout=30,
        )
        dt = (time.time() - t0) * 1000
        if r.ok:
            js = r.json()
            st.success(f"{js['label']} (conf {js['confidence']:.2f}) — {dt:.0f} ms")
            st.json(js)
        else:
            st.error(f"{r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"Predict failed: {e}")
