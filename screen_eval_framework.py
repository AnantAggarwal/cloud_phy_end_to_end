import os
import cv2
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from ultralytics import YOLO  # works for YOLOv8 & YOLOv11


# ---------- Base Classes ---------- #

class ScreenSegmenter:
    def segment(self, image):
        """Return segmented/cleaned version of the screen image"""
        raise NotImplementedError


class Localiser:
    def localise(self, image):
        """Return list of bounding boxes [(x1, y1, x2, y2, label), ...]"""
        raise NotImplementedError


class OCRModel:
    def read(self, image, boxes):
        """Return dict[label -> recognised_text]"""
        raise NotImplementedError


# ---------- YOLO-based Implementations ---------- #

class YOLOv8Segmenter(ScreenSegmenter):
    """Uses YOLOv8 to segment out screen region (cropping largest box)."""

    def __init__(self, model_path: str, conf: float = 0.5, device: str = "cuda"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def segment(self, image):
        results = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return image  # fallback: no segmentation
        # choose largest bbox (screen region)
        areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
        idx = int(max(range(len(areas)), key=lambda i: areas[i]))
        x1, y1, x2, y2 = boxes[idx]
        seg = image[int(y1):int(y2), int(x1):int(x2)]
        return seg


class YOLOv11Localiser(Localiser):
    """Uses YOLOv11 to detect field regions inside screen."""

    def __init__(self, model_path: str, conf: float = 0.5, device: str = "cuda"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def localise(self, image) -> List[tuple]:
        results = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)
        det = results[0]
        boxes = det.boxes.xyxy.cpu().numpy()
        labels = det.names
        classes = det.boxes.cls.cpu().numpy().astype(int)

        bboxes = []
        for (x1, y1, x2, y2, cls_id) in zip(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], classes):
            label = labels[cls_id] if cls_id in labels else f"class_{cls_id}"
            bboxes.append((int(x1), int(y1), int(x2), int(y2), label))
        return bboxes


# ---------- Example OCR Implementation (PaddleOCR) ---------- #

class PaddleOCROnly(OCRModel):
    """Simple OCR model using PaddleOCR in recognition-only mode."""

    def __init__(self, use_gpu=False):
        from paddleocr import PaddleOCR
        self.model = PaddleOCR(det=False, lang='en')

    def read(self, image, boxes):
        results = {}
        for (x1, y1, x2, y2, label) in boxes:
            crop = image[y1:y2, x1:x2]
            rec = self.model.ocr(crop, det=False, rec=True)
            text = rec[0][0][0] if rec and rec[0] else ""
            results[label] = text
        return results


# ---------- Core Evaluator ---------- #

class ScreenEvaluator:
    def __init__(self, segment_model: ScreenSegmenter,
                 localise_model: Localiser,
                 ocr_model: OCRModel):
        self.segment_model = segment_model
        self.localise_model = localise_model
        self.ocr_model = ocr_model

    def process_image(self, img_path: str) -> Dict[str, Any]:
        image = cv2.imread(img_path)
        if image is None:
            return {"image": img_path, "error": "cannot read"}

        result = {"image": img_path}
        t0 = time.time()
        try:
            # Step 1: Screen segmentation
            seg = self.segment_model.segment(image)

            # Step 2: Localisation
            boxes = self.localise_model.localise(seg)

            # Step 3: OCR
            ocr_out = self.ocr_model.read(seg, boxes)
            result.update(ocr_out)
            result["num_boxes"] = len(boxes)
        except Exception as e:
            result["error"] = str(e)

        result["time_sec"] = round(time.time() - t0, 3)
        return result

    def run_on_directory(self, image_dir: str, output_csv: str):
        paths = sorted(list(Path(image_dir).glob("*.jpg")) +
                       list(Path(image_dir).glob("*.png")))
        rows = []
        for i, p in enumerate(paths):
            out = self.process_image(str(p))
            rows.append(out)
            print(f"[{i+1}/{len(paths)}] {p.name} done in {out['time_sec']}s")
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Results saved to {output_csv}")


# ---------- Example Run ---------- #
if __name__ == "__main__":
    # Paths to your trained weights
    SEGMENT_MODEL_PATH = "/kaggle/input/yolov8-screen-segmentation/best.pt"
    LOCALISE_MODEL_PATH = "/kaggle/input/yolov11-localisation/best.pt"

    segment_model = YOLOv8Segmenter(model_path=SEGMENT_MODEL_PATH, conf=0.5, device="cuda")
    localise_model = YOLOv11Localiser(model_path=LOCALISE_MODEL_PATH, conf=0.5, device="cuda")
    ocr_model = PaddleOCROnly(use_gpu=True)

    evaluator = ScreenEvaluator(segment_model, localise_model, ocr_model)
    evaluator.run_on_directory(
        image_dir="/kaggle/input/segmented-screens", 
        output_csv="/kaggle/working/eval_results.csv"
    )
