"""
This script filters out non-chick pixels from an input image or video and saves the result.
"""

import cv2
import numpy as np

# ——— CONFIG —————————————————————————————————————————————————
IN_PATH         = ""
OUT_PATH        = ""
PROCESS_VIDEO   = True    # true for video, false for image
# ————————————————————————————————————————————————————————————

def process_frame(frame):
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	r = frame_rgb[:, :, 0].astype(np.float32)
	g = frame_rgb[:, :, 1].astype(np.float32)
	b = frame_rgb[:, :, 2].astype(np.float32)

	rg_diff = np.abs(r - g)
	rg_avg_minus_b = (r + g) / 2 - b
	min_rg = np.minimum(r, g)

	chick_mask = ((rg_diff <= 16) & 
					(rg_avg_minus_b >= 20) & 
					(min_rg >= 80) &
					(r > b) & 
					(g > b))

	filtered = np.zeros_like(frame_rgb)
	filtered[chick_mask] = frame_rgb[chick_mask]

	return cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR)

if PROCESS_VIDEO:
	cap = cv2.VideoCapture(IN_PATH)
	assert cap.isOpened(), "Could not open input video"

	w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS)

	fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
	writer_vid = cv2.VideoWriter(OUT_PATH, fourcc, fps, (w, h))
	assert writer_vid.isOpened(), "Could not open output video"

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		filtered_frame = process_frame(frame)
		writer_vid.write(filtered_frame)

	cap.release()
	writer_vid.release()
	print(f"Video saved to {OUT_PATH}")
else:
	frame = cv2.imread(IN_PATH)
	assert frame.isOpened(), "Could not open input image"

	filtered_frame = process_frame(frame)
	cv2.imwrite(OUT_PATH, filtered_frame)
	print(f"Image saved to {OUT_PATH}")