import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import prometheus_client as prom
import time, zipfile, io, socket

# Force CPU — MPS causes segfault with BLIP on M1
DEVICE = torch.device("cpu")

# ── Registry-safe metric helpers ──────────────────────────────────────────────
def _counter(name, desc, labels=[]):
    try:
        return prom.Counter(name, desc, labels)
    except ValueError:
        return prom.REGISTRY._names_to_collectors.get(name + "_total") \
            or prom.REGISTRY._names_to_collectors.get(name)

def _gauge(name, desc):
    try:
        return prom.Gauge(name, desc)
    except ValueError:
        return prom.REGISTRY._names_to_collectors.get(name)

def _histogram(name, desc, labels=[], buckets=None):
    try:
        return prom.Histogram(name, desc, labels,
               buckets=buckets or prom.DEFAULT_BUCKETS)
    except ValueError:
        return prom.REGISTRY._names_to_collectors.get(name)

def _summary(name, desc):
    try:
        return prom.Summary(name, desc)
    except ValueError:
        return prom.REGISTRY._names_to_collectors.get(name)

# ── Metrics ───────────────────────────────────────────────────────────────────
images_processed   = _counter("captioning_images_processed_total", "Total images processed", ["mode"])
errors_total       = _counter("captioning_errors_total", "Total errors", ["error_type"])
requests_by_ip     = _counter("captioning_requests_by_source_total", "Requests by IP", ["source_ip", "mode"])
active_requests    = _gauge("captioning_active_requests", "In-flight requests")
model_memory_mb    = _gauge("captioning_model_memory_mb", "Model memory MB")
inference_latency  = _histogram("captioning_inference_latency_seconds", "Inference time", ["mode"], [0.5, 1, 2, 5, 10, 30])
bulk_batch_size    = _histogram("captioning_bulk_batch_size", "Images per ZIP", [], [1, 5, 10, 20, 50, 100])
caption_word_count = _summary("captioning_caption_word_count", "Caption word count")

# ── Start metrics server ONCE ─────────────────────────────────────────────────
if "metrics_started" not in st.session_state:
    try:
        prom.start_http_server(8001)
    except OSError:
        pass
    st.session_state["metrics_started"] = True

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    model_memory_mb.set(900)
    return processor, model

# ── Caption function ──────────────────────────────────────────────────────────
def caption_image(image, mode, source_ip):
    active_requests.inc()
    t0 = time.time()
    try:
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        elapsed = time.time() - t0
        inference_latency.labels(mode=mode).observe(elapsed)
        images_processed.labels(mode=mode).inc()
        requests_by_ip.labels(source_ip=source_ip, mode=mode).inc()
        caption_word_count.observe(len(caption.split()))
        return caption, elapsed
    except Exception as e:
        errors_total.labels(error_type=type(e).__name__).inc()
        raise
    finally:
        active_requests.dec()

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Image Captioning", layout="centered")
st.title("AI Image Captioning")
st.caption(f"Device: `{DEVICE}` · Metrics: `http://localhost:8001/metrics`")

processor, model = load_model()
source_ip = socket.gethostbyname(socket.gethostname())
mode = st.radio("Mode", ["Single image", "Bulk ZIP"], horizontal=True)

if mode == "Single image":
    f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if f:
        img = Image.open(f).convert("RGB")
        st.image(img, use_container_width=True)
        with st.spinner("Generating caption..."):
            cap, elapsed = caption_image(img, "single", source_ip)
        st.success(f"**Caption:** {cap}")
        st.caption(f"Inference time: {elapsed:.2f}s")

else:
    f = st.file_uploader("Upload a ZIP of images", type=["zip"])
    if f:
        with zipfile.ZipFile(io.BytesIO(f.read())) as z:
            names = [n for n in z.namelist()
                     if n.lower().endswith((".jpg",".jpeg",".png"))
                     and not n.startswith("__MACOSX")]
            if not names:
                st.error("No images found in ZIP.")
            else:
                bulk_batch_size.observe(len(names))
                st.info(f"Found {len(names)} images — processing...")
                progress = st.progress(0)
                results = []
                for i, name in enumerate(names):
                    with z.open(name) as img_file:
                        img = Image.open(img_file).convert("RGB")
                        cap, elapsed = caption_image(img, "bulk", source_ip)
                        results.append((name, img, cap, elapsed))
                    progress.progress((i + 1) / len(names))
                st.success(f"Done! Captioned {len(names)} images.")
                st.divider()
                for name, img, cap, elapsed in results:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(img, use_container_width=True)
                    with col2:
                        st.markdown(f"**{name}**")
                        st.write(cap)
                        st.caption(f"{elapsed:.2f}s")
                    st.divider()
