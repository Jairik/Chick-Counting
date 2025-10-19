import cv2, pytesseract, re

def ocr_number(img):
    cfg = r'--psm 7 -c tessedit_char_whitelist=0123456789.-'
    text = pytesseract.image_to_string(img, config=cfg)
    m = re.search(r'-?\d+(?:\.\d+)?', text)
    return float(m.group()) if m else None

cap = cv2.VideoCapture("thermal.mp4")
ok, frame0 = cap.read()
assert ok

# Define ROIs in (x,y,w,h). Do this once; adjust to your overlay.
ROI_HIGH = (x1, y1, w1, h1)
ROI_LOW  = (x2, y2, w2, h2)

results = []
frame_idx = 0
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

while True:
    ok, frame = cap.read()
    if not ok: break
    x,y,w,h = ROI_HIGH
    hi_roi = frame[y:y+h, x:x+w]
    x,y,w,h = ROI_LOW
    lo_roi = frame[y:y+h, x:x+w]

    # Preprocess
    def prep(roi):
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3,3), 0)
        g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        return g

    hi_val = ocr_number(prep(hi_roi))
    lo_val = ocr_number(prep(lo_roi))

    t = frame_idx / fps
    results.append((frame_idx, t, hi_val, lo_val))
    frame_idx += 1

cap.release()
