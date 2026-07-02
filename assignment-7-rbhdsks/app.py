import io

import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps

MLFLOW_ENDPOINT = "http://127.0.0.1:5001/invocations"

st.set_page_config(page_title="MNIST Classifier", page_icon="🔢")
st.title("🔢 MNIST Digit Classifier")
st.caption("Upload a digit image — it gets sent to the MLflow-served model and classified.")

uploaded_file = st.file_uploader(
    "Upload an image (PNG/JPG). Works best with a clean digit on a light background.",
    type=["png", "jpg", "jpeg"],
)


def preprocess(pil_image: Image.Image) -> np.ndarray:
    """Convert an uploaded image into the [1, 1, 28, 28] float32 tensor the model expects."""
    img = pil_image.convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32)
    # Auto-detect: if the background (mean) is dark, it's already MNIST-style (white on black)
    # If the background is light, invert it to match MNIST format
    if arr.mean() > 127:
        arr = 255.0 - arr
    arr = arr / 255.0
    arr = (arr - 0.1307) / 0.3081
    arr = arr.reshape(1, 1, 28, 28)
    return arr

def predict(tensor: np.ndarray):
    payload = {"inputs": tensor.tolist()}
    response = requests.post(MLFLOW_ENDPOINT, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Preprocessed (28×28)")
        preprocessed = preprocess(image)
        # Undo normalization just for display
        display_img = (preprocessed[0, 0] * 0.3081 + 0.1307) * 255
        display_img = np.clip(display_img, 0, 255).astype(np.uint8)
        st.image(display_img, use_container_width=True, clamp=True)

    if st.button("🚀 Classify", type="primary"):
        with st.spinner("Calling MLflow model server..."):
            try:
                result = predict(preprocessed)
                logits = np.array(result["predictions"][0])
                # Softmax for probabilities
                exp_logits = np.exp(logits - logits.max())
                probs = exp_logits / exp_logits.sum()
                predicted_digit = int(np.argmax(probs))
                confidence = float(probs[predicted_digit])

                st.success(f"**Predicted digit: {predicted_digit}**  (confidence: {confidence:.2%})")

                st.subheader("Class probabilities")
                st.bar_chart({"probability": probs.tolist()})
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the MLflow model server at "
                    f"`{MLFLOW_ENDPOINT}`. Make sure it's running:\n\n"
                    "```\nmlflow models serve -m runs:/<run_id>/model --port 5001 --env-manager=local\n```"
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.info("👆 Upload a digit image to get started.")

st.divider()
st.caption(f"Model endpoint: `{MLFLOW_ENDPOINT}`")