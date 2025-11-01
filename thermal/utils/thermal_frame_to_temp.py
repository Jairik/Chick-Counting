''' Helper functionality to return the temperature data of an individual thermal frame (to use in bound box validation pipeline) '''
import cv2, pytesseract, re, joblib
from typing import Literal
import blosc
import numpy as np
import matplotlib.pyplot as plt
from ultralytics.engine.results import Results

__all__ = ['result_to_temp_frame']

# OCR helpers
def ocr_number(img):
    cfg = r'--psm 7 -c tessedit_char_whitelist=0123456789.-'
    text = pytesseract.image_to_string(img, config=cfg)
    m = re.search(r'-?\d+(?:\.\d+)?', text)
    return float(m.group()) if m else None
def prep(roi):
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return g

# Helper to pull a BGR ndarray from a YOLO Results object
def _extract_bgr_from_yolo_res(yolo_res) -> np.ndarray:
    if hasattr(yolo_res, "orig_img") and isinstance(yolo_res.orig_img, np.ndarray):
        return yolo_res.orig_img
    if hasattr(yolo_res, "imgs"):
        img = yolo_res.imgs[0] if isinstance(yolo_res.imgs, (list, tuple)) else yolo_res.imgs
        if isinstance(img, np.ndarray):
            return img
    if hasattr(yolo_res, "plot"):
        plotted = yolo_res.plot()
        if isinstance(plotted, np.ndarray):
            return plotted
    raise ValueError("Could not extract frame ndarray from YOLO Results object.")

# Main functionality (slightly modifying the lovely code written by Logan to process each frame individually)
def result_to_temp_frame(
        result: Results,
        frame_idx: int = 0,
        fps: float = 30.0,
        prev_hi_val: float | None = None,
        prev_lo_val: float | None = None
    ) -> tuple[np.ndarray, float, float]:
    ''' Given a YOLO results object, will denormalize the thermal image and extract the raw temperature data '''
    
    # Fixed location of the heatbar
    x1 = 190
    y1 = 57
    y2 = 240
    w1 = 60
    h1 = 18

    # Defining ROIs in (x,y,w,h)
    ROI_HIGH = (x1, y1, w1, h1)
    ROI_LOW  = (x1, y2, w1, h1)

    # Getting the BGR frame from the YOLO result
    frame = _extract_bgr_from_yolo_res(result)
    t = frame_idx / (fps or 30.0)

    # Only run OCR every four frames; otherwise carry forward previous values
    if (frame_idx % 4) != 1:
        if prev_hi_val is None or prev_lo_val is None:
            raise ValueError("prev_hi_val/prev_lo_val required for non-OCR frames (frame_idx % 4 > 0).")
        hi_val = prev_hi_val
        lo_val = prev_lo_val
    else:
        # declare windows for ocr
        x,y,w,h = ROI_HIGH
        hi_roi = frame[y:y+h, x:x+w]
        x,y,w,h = ROI_LOW
        lo_roi = frame[y:y+h, x:x+w]

        # attempt detection of numbers in these windows
        hi_val = ocr_number(prep(hi_roi))
        lo_val = ocr_number(prep(lo_roi))

        # if OCR missed but we have previous → reuse
        if hi_val is None and prev_hi_val is not None:
            hi_val = prev_hi_val
        if lo_val is None and prev_lo_val is not None:
            lo_val = prev_lo_val

    if hi_val is None or lo_val is None:
        raise ValueError(f"OCR failed for hi/lo temps. Got (hi, lo)=({hi_val}, {lo_val}).")

    # if OCR gave us flipped values (like hi=2.4, lo=15.7) → swap them
    if hi_val < lo_val:
        hi_val, lo_val = lo_val, hi_val

    # Build a temp table
    temp_rngs = np.asarray([(frame_idx, t, hi_val, lo_val)], dtype=object)
    if temp_rngs.shape[0] > 1:
        for m in range(1, temp_rngs.shape[0]):
            if (temp_rngs[m, 2] < 55 or temp_rngs[m, 2] > 62):
                temp_rngs[m, 2] = temp_rngs[m-1, 2]
            if (temp_rngs[m, 3] < 14 or temp_rngs[m, 3] > 17):
                temp_rngs[m, 3] = temp_rngs[m-1, 3]
        if temp_rngs.shape[0] > 8:
            temp_rngs[:8, 2:] = temp_rngs[8, 2:]

    # Denormalization logic
    norm_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    norm_frame = norm_frame.astype(np.float32) / 255  # scale val range [0, 255] to [0.0, 1.0]

    # Retreive the high/low values computer earlier
    lo_t = float(temp_rngs[0, 3])
    hi_t = float(temp_rngs[0, 2])
    df_t = hi_t - lo_t  # Getting the value range for proper scaling
    
    if df_t < 0:
        # as a last resort, if we still got a negative diff but have prevs, reuse them
        if prev_hi_val is not None and prev_lo_val is not None:
            hi_t = float(prev_hi_val)
            lo_t = float(prev_lo_val)
            df_t = hi_t - lo_t
        if df_t < 0:
            raise ValueError(f"negative hi-lo temp difference, impossible. GOT (hi, lo)=({hi_val}, {lo_val})")
    
    # Scale the matrix by the real temp range
    norm_frame *= df_t
    # Displace the values of the scaled matrix to sit in the real temperature values 
    norm_frame += lo_t

    return norm_frame, float(hi_val), float(lo_val)  # Return the raw temperature array for this frame, plus the hi/lo we used
