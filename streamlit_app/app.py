import os
import time
import requests
import streamlit as st
from urllib.parse import urljoin

st.set_page_config(page_title="Deepfake Detector Demo", layout="centered")
st.title("Deepfake Detector — Streamlit Demo")

# ---- API base resolution -----------------------------------------------------
# Priority: API_BASE env → In-cluster DNS → NodePort fallback
def default_api_base() -> str:
    env = os.getenv("API_BASE")
    if env:
        return env.strip().rstrip("/")
    # Heuristic: if running in k8s, the cluster DNS is best
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return "http://deepfake-api-svc.deepfake.svc.cluster.local"
    # Local dev fallback (NodePort via minikube tunnel/port-forward)
    node_ip = os.getenv("NODE_IP", st.session_state.get("node_ip", "127.0.0.1"))
    node_port = os.getenv("NODE_PORT", st.session_state.get("node_port", "30080"))
    return f"http://{node_ip}:{node_port}"

api_url = st.text_input("API base URL", value=default_api_base()).strip().rstrip("/")

def _safe_get(url: str, timeout=5):
    return requests.get(url, timeout=timeout)

def _safe_post(url: str, files: dict, timeout=30):
    return requests.post(url, files=files, timeout=timeout)

# Validate URL early
if api_url:
    try:
        r = _safe_get(urljoin(api_url + "/", "healthz"), timeout=3)
        st.caption(f"Health ping: {r.status_code}")
    except Exception as e:
        st.caption(f"Health ping failed: {e}")

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])

col1, col2 = st.columns(2)
with col1:
    if st.button("Health"):
        if not api_url:
            st.error("API base URL is empty.")
        else:
            try:
                r = _safe_get(urljoin(api_url + "/", "healthz"), timeout=5)
                # healthz may be json or text; handle both
                try:
                    st.code(r.json())
                except Exception:
                    st.text(r.text)
            except Exception as e:
                st.error(f"Health request failed: {e}")

with col2:
    if st.button("Metrics (head)"):
        if not api_url:
            st.error("API base URL is empty.")
        else:
            try:
                r = _safe_get(urljoin(api_url + "/", "metrics"), timeout=5)
                st.text("\n".join(r.text.splitlines()[:15]))
            except Exception as e:
                st.error(f"Metrics request failed: {e}")

st.markdown("---")

if st.button("Predict"):
    if not file:
        st.warning("Please upload an image first.")
    elif not api_url:
        st.error("API base URL is empty.")
    else:
        try:
            t0 = time.time()
            r = _safe_post(
                urljoin(api_url + "/", "predict"),
                files={"file": (file.name, file.getvalue(), file.type or "image/jpeg")},
                timeout=30,
            )
            dt = (time.time() - t0) * 1000
            if r.ok:
                js = r.json()
                label = js.get("label", "unknown")
                conf = js.get("confidence", 0.0)
                st.success(f"{label} (conf {conf:.2f}) — {dt:.0f} ms")
                st.json(js)
            else:
                st.error(f"{r.status_code}: {r.text[:500]}")
        except Exception as e:
            st.error(f"Predict failed: {e}")