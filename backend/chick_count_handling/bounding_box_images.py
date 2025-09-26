"""
This script crops frames from raw video using pre-logged detections.
Inputs:
	- recorded chick video
	- spreadsheet outputted from bounding_box_data.py using said recorded chick video
Outputs:
	- folder of cropped images containing just bounding box detections
Purpose:
	- obtain said images without counting line and bounding box line obstructions
	- used to train for future color segmenting
"""

import os
import cv2
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────
VIDEO_PATH        = ""
CROSSINGS_CSV     = ""
SNAPSHOT_DIR      = "" #include the folder you would like to save the images at
# ───────────────────────────────────────────────────────────────────

def crop_exact(img, box):
	x1, y1, x2, y2 = map(int, box)
	return img[y1:y2, x1:x2]

def main():
	os.makedirs(SNAPSHOT_DIR, exist_ok=True)
	df = pd.read_csv(CROSSINGS_CSV, usecols=["frame", "main_id", "x1", "y1", "x2", "y2"])
	cap = cv2.VideoCapture(VIDEO_PATH)
	assert cap.isOpened(), f"Could not open {VIDEO_PATH}"

	for _, row in df.iterrows():
		frame = row["frame"]
		tid = row["main_id"]
		box = (row["x1"], row["y1"], row["x2"], row["y2"])

		cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
		ret, img = cap.read()
		if not ret:
			print(f"Frame {frame} not readable, skipped.")
			continue

		crop = crop_exact(img, box)
		out_name = f"frame{frame:06d}_id{tid}.png"
		out_path = os.path.join(SNAPSHOT_DIR, out_name)
		cv2.imwrite(out_path, crop)
		print(f"frame{frame:06d}_id{tid}.png saved")

	cap.release()
	print(f"Done. Snapshots saved in: {SNAPSHOT_DIR}")

if __name__ == "__main__":
	main()