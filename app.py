import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR
import torch

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

CONF_THRESHOLD = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.05, 
    max_value=1.0, 
    value=0.5, 
    step=0.05
)


# ---- Model Initialization ----
@st.cache_resource
def load_models(segment_path, localise_path, conf_threshold, use_gpu=True):
    """Load models once and cache them."""
    
    # Check if GPU is requested AND available
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        gpu_for_paddle = True
        st.sidebar.success("CUDA (GPU) device found!")
    else:
        device = "cpu"
        gpu_for_paddle = False
        if use_gpu: # If user wanted GPU but it's not available
            st.sidebar.warning("CUDA not available. Falling back to CPU.")
        
    segment_model = YOLOv8Segmenter(model_path=segment_path, conf=conf_threshold, device=device)
    localise_model = YOLOv11Localiser(model_path=localise_path, conf=conf_threshold, device=device)
    ocr_model = PaddleOCROnly(use_gpu=gpu_for_paddle)
    
    return segment_model, localise_model, ocr_model


segment_model, localise_model, ocr_model = load_models(
    SEGMENT_MODEL_PATH, LOCALISE_MODEL_PATH, CONF_THRESHOLD, USE_GPU
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

        # --- Create visualization for Localisation ONLY ---
        vis_boxes_only = segmented.copy()
        for (x1, y1, x2, y2, label) in boxes:
            cv2.rectangle(vis_boxes_only, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_boxes_only, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # ----------------------------------------------------

        # Step 3: OCR
        ocr_results = ocr_model.read(segmented, boxes)

        # --- Create Combined Visualization (Boxes + OCR Text) ---
        vis_combined = segmented.copy()
        for (x1, y1, x2, y2, label) in boxes:
            text = ocr_results.get(label, "")
            cv2.rectangle(vis_combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_combined, f"{label}: {text}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # --------------------------------------------------------

    # ---- Display results ----
    st.subheader("🔍 Step 1: Segmented Screen")
    st.image(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("📍 Step 2: Localisation Results (Boxes)")
    st.image(cv2.cvtColor(vis_boxes_only, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("🧾 Step 3: OCR Results (Table)")
    if ocr_results:
        df = pd.DataFrame(list(ocr_results.items()), columns=["Field", "Value"])
        st.table(df)
        # Option to download CSV
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Results as CSV", csv_data, "results.csv", "text/csv")
    else:
        st.warning("No readable text detected.")

    st.subheader("✅ Combined Result (Localisation + OCR)")
    st.image(cv2.cvtColor(vis_combined, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    st.success("✅ Processing complete!")

else:
    st.info("Upload an image above to begin.")