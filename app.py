import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ---- Import components from your framework ----
from screen_eval_framework import YOLOv8Segmenter, YOLOv11Localiser, PaddleOCROnly


# ---- Streamlit setup ----
st.set_page_config(page_title="Screen Value Extractor", page_icon="🩺", layout="wide")
st.title("🩺 Screen Value Extraction Demo")
st.markdown("Upload a segmented screen image and let the app detect and extract vital readings automatically.")

# ---- Sidebar Configuration ----
st.sidebar.header("Model Configuration")

SEGMENT_MODEL_PATH = st.sidebar.text_input(
    "YOLOv8 Segmenter Model Path (.pt)",
    "yolo8.pt"
)
LOCALISE_MODEL_PATH = st.sidebar.text_input(
    "YOLOv11 Localiser Model Path (.pt)",
    "yolo11.pt"
)
USE_GPU = st.sidebar.checkbox("Use GPU (if available)", True)


# ---- Model Initialization ----
@st.cache_resource
def load_models(segment_path, localise_path, use_gpu=True):
    """Load models once and cache them."""
    device = "cuda" if use_gpu else "cpu"
    segment_model = YOLOv8Segmenter(model_path=segment_path, conf=0.5, device=device)
    localise_model = YOLOv11Localiser(model_path=localise_path, conf=0.5, device=device)
    ocr_model = PaddleOCROnly(use_gpu=use_gpu)
    return segment_model, localise_model, ocr_model


segment_model, localise_model, ocr_model = load_models(
    SEGMENT_MODEL_PATH, LOCALISE_MODEL_PATH, USE_GPU
)


# ---- File Upload ----
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)[:, :, ::-1]  # Convert RGB → BGR for OpenCV
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running segmentation, localisation, and OCR..."):
        # Step 1: Screen Segmentation
        segmented = segment_model.segment(image_np)

        # Step 2: Localisation
        boxes = localise_model.localise(segmented)

        # Step 3: OCR
        ocr_results = ocr_model.read(segmented, boxes)

        # Visualization
        vis = segmented.copy()
        for (x1, y1, x2, y2, label) in boxes:
            text = ocr_results.get(label, "")
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{label}: {text}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---- Display results ----
    st.subheader("🔍 Segmented Screen")
    st.image(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("📍 Localisation + OCR Results")
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("🧾 Extracted Values")
    if ocr_results:
        df = pd.DataFrame(list(ocr_results.items()), columns=["Field", "Value"])
        st.table(df)
        # Option to download CSV
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Results as CSV", csv_data, "results.csv", "text/csv")
    else:
        st.warning("No readable text detected.")

    st.success("✅ Processing complete!")

else:
    st.info("Upload an image above to begin.")
