"""
This script creates a .csv of RGB value counts with binning.

Input
- (N, 3) uint8 .npy file with rows = [R, G, B]
Output
- file with columns: red_value, green_value, blue_value, count

Parameters
- acceptable bins include: 1, 2, 4, 8, 16
	+ bins group nearby RGB values into ranges of size BIN_SIZE
	+ reduces color detail so similar shades are counted together
- CHUNK_ROWS -> how many pixels to process per loop, controls temp memory

Used to pull color value data for training color segmentation for RGB model.
""" 

import os
import numpy as np

# ——— CONFIG —————————————————————————————————————————————————
NPY_PATH         = "c:/Users/anye forti/Desktop/2025 FALL/426 COSC/YOLO_TESTING/Chick-Counting/data/pixel_rgb_values/rgb_values_v2.npy"
OUT_CSV          = "c:/Users/anye forti/Desktop/2025 SPRING/425 COSC/YOLO_TESTING/Chick-Counting/data/pixel_rgb_values/rgb_values_v2_count_bin1.csv"
BIN_SIZE         = 1
CHUNK_ROWS       = 250000
# ————————————————————————————————————————————————————————————

ALLOWED_BINS = {1, 2, 4, 8, 16}

if BIN_SIZE not in ALLOWED_BINS:
	print("bins must be one of these: 1, 2, 4, 8, 16")
	raise SystemExit(1)

arr = np.load(NPY_PATH, mmap_mode="r")
nR = nG = nB = 256 // BIN_SIZE
total_bins = nR * nG * nB
counts = np.zeros(total_bins, dtype=np.uint32)

N = len(arr)
nGB = nG * nB

for start in range(0, N, CHUNK_ROWS):
	end = min(start + CHUNK_ROWS, N)
	block = arr[start:end]

	r_idx = (block[:, 0] // BIN_SIZE).astype(np.uint32)
	g_idx = (block[:, 1] // BIN_SIZE).astype(np.uint32)
	b_idx = (block[:, 2] // BIN_SIZE).astype(np.uint32)

	keys = r_idx * nGB + g_idx * nB + b_idx
	bc = np.bincount(keys)
	counts[:bc.size] += bc.astype(np.uint32)

nonzero = np.nonzero(counts)[0]
unique_bins = len(nonzero)

os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    f.write("red_value,green_value,blue_value,count\n")
    for key in nonzero:
        ct = int(counts[key])

        r_bin = int(key // nGB)
        rem   = int(key %  nGB)
        g_bin = int(rem // nB)
        b_bin = int(rem %  nB)

        r_val = r_bin * BIN_SIZE
        g_val = g_bin * BIN_SIZE
        b_val = b_bin * BIN_SIZE

        f.write(f"{r_val},{g_val},{b_val},{ct}\n")

print(f"Saved .csv to {OUT_CSV}, unique bins: {unique_bins}, bin={BIN_SIZE}")