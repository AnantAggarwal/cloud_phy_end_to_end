# screen_eval_framework.py

import os
import cv2
import json
import time
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, Any, List, Tuple

# ---------- Interfaces (abstract stage signatures) ---------- #

class ScreenSegmenter:
    def segment(self, image):
        """Return a segmented/cleaned version of the screen image"""
        raise NotImplementedError

class Localiser:
    def localise(self, image):
        """Return list of bounding boxes [(x1, y1, x2, y2, label), ...]"""
        raise NotImplementedError

class OCRModel:
    def read(self, image, boxes):
        """Return dict[label -> recognised_text]"""
        raise NotImplementedError


# ---------- Example dummy implementations ---------- #
# You can replace these later with your real models

class IdentitySegmenter(ScreenSegmenter):
    def segment(self, image): 
        return image  # no change

class DummyLocaliser(Localiser):
    def localise(self, image):
        h, w = image.shape[:2]
        return [(0, 0, w, h, 'full_screen')]

class TesseractOCR(OCRModel):
    def __init__(self):
        import pytesseract
        self.pytesseract = pytesseract
    def read(self, image, boxes):
        results = {}
        for (x1,y1,x2,y2,label) in boxes:
            crop = image[y1:y2, x1:x2]
            txt = self.pytesseract.image_to_string(crop).strip()
            results[label] = txt
        return results


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
            seg = self.segment_model.segment(image)

            boxes = self.localise_model.localise(seg)

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
        print(f"\nSaved results → {output_csv}")


if __name__ == "__main__":
    segment_model = IdentitySegmenter()
    localise_model = DummyLocaliser()
    ocr_model = TesseractOCR()

    evaluator = ScreenEvaluator(segment_model, localise_model, ocr_model)
    evaluator.run_on_directory(
        image_dir="/kaggle/input/segmented-screens", 
        output_csv="/kaggle/working/results.csv"
    )
