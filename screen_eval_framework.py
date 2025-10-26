import os
import cv2
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import torch  # <-- ADD THIS LINE
import numpy as np

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
    """
    Uses YOLOv8 OBB (Oriented Bounding Box) to segment out screen region.
    It finds the largest detected OBB and performs a perspective warp
    to return a straightened, rectangular crop of the screen.
    """

    def __init__(self, model_path: str, conf: float = 0.5, device: str = "cuda"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        print(f"YOLOv8Segmenter (OBB): Loaded {model_path} on {device} with conf={conf}")

    def segment(self, image):
        results = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)
        
        # Check if the 'obb' attribute exists and has results
        if results[0].obb is None:
            print(f"YOLOv8Segmenter: No OBB results found (results[0].obb is None) at conf={self.conf}. Returning original image.")
            return image

        obbs = results[0].obb
        
        # Check if any boxes were detected
        if len(obbs) == 0:
            print(f"YOLOv8Segmenter: No boxes found (len(obbs) == 0) at conf={self.conf}. Returning original image.")
            return image

        try:
            # --- HELPER FUNCTION TO SORT CORNERS ---
            def order_points(pts):
                # initialzie a list of coordinates that will be ordered
                # such that the first entry in the list is the top-left,
                # the second entry is the top-right, the third is the
                # bottom-right, and the fourth is the bottom-left
                rect = np.zeros((4, 2), dtype="float32")
                
                # the top-left point will have the smallest sum, whereas
                # the bottom-right point will have the largest sum
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                
                # now, compute the difference between the points, the
                # top-right point will have the smallest difference,
                # whereas the bottom-left will have the largest difference
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                
                # return the ordered coordinates
                return rect
            # ----------------------------------------

            # Get the 4 corner points for the largest box
            corners = obbs.xyxyxyxy.cpu().numpy()[idx]

            # Order the points into [tl, tr, br, bl]
            src_pts = order_points(corners.astype(np.float32))

            # Unpack the ordered points
            (tl, tr, br, bl) = src_pts

            # --- Calculate width and height from the ORDERED points ---
            # Calculate the width of the new image (max of top/bottom edges)
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            # Calculate the height of the new image (max of left/right edges)
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            # -----------------------------------------------------------------
            
            # Define the destination points for the perspective warp
            # This order MUST match the `order_points` output: [tl, tr, br, bl]
            dst_pts = np.array([
                [0, 0],                     # Top-left
                [maxWidth - 1, 0],          # Top-right
                [maxWidth - 1, maxHeight - 1], # Bottom-right
                [0, maxHeight - 1]          # Bottom-left
            ], dtype=np.float32)

            # Calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Apply the warp
            warped_img = cv2.warpPerspective(image, M, (int(maxWidth), int(maxHeight)))
            
            print(f"YOLOv8Segmenter: Successfully warped OBB to {warped_img.shape[:2]}.")
            return warped_img

        except Exception as e:
            print(f"YOLOv8Segmenter: Error during OBB processing: {e}. Returning original image.")
            # Fallback to simple crop
            try:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
                    idx = int(max(range(len(areas)), key=lambda i: areas[i]))
                    x1, y1, x2, y2 = boxes[idx]
                    seg = image[int(y1):int(y2), int(x1):int(x2)]
                    print("YOLOv8Segmenter: Fallback to simple AABB crop.")
                    return seg
                else:
                    return image
            except Exception:
                 return image # Final fallback
class YOLOv11Localiser(Localiser):
    """Uses YOLOv11 to detect field regions inside screen."""

    def __init__(self, model_path: str, conf: float = 0.5, device: str = "cuda"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def localise(self, image) -> List[tuple]:
        results = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)
        print(results)
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
        # Initialize the model once
        self.model = PaddleOCR(lang='en', use_gpu=use_gpu)

    def read(self, image, boxes):
        results = {}
        for (x1, y1, x2, y2, label) in boxes:
            # Crop the image to the bounding box
            crop = image[y1:y2, x1:x2]
            
            # --- FIX: Removed the 'cls=False' argument ---
            rec = self.model.ocr(crop)
            # ---------------------------------------------

            # Extract text from the result structure
            text = ""
            if rec and rec[0]:
                text = rec[0][0][0] # Get the text from the first result
                
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
    
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_gpu_for_paddle = True if device == "cuda" else False
    print(f"Running models on device: {device}")

    # Paths to your trained weights
    SEGMENT_MODEL_PATH = "/kaggle/input/yolov8-screen-segmentation/best.pt"
    LOCALISE_MODEL_PATH = "/kaggle/input/yolov11-localisation/best.pt"

    segment_model = YOLOv8Segmenter(model_path=SEGMENT_MODEL_PATH, conf=0.5, device=device)
    localise_model = YOLOv11Localiser(model_path=LOCALISE_MODEL_PATH, conf=0.5, device=device)
    ocr_model = PaddleOCROnly(use_gpu=use_gpu_for_paddle)

    evaluator = ScreenEvaluator(segment_model, localise_model, ocr_model)
    evaluator.run_on_directory(
        image_dir="/kaggle/input/segmented-screens", 
        output_csv="/kaggle/working/eval_results.csv"
    )
