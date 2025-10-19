"""
This script filters out non-chick pixels from an input image and saves the result.
"""

import cv2
import numpy as np

# ——— CONFIG —————————————————————————————————————————————————
IN_IMG         = r"C:\Users\anye forti\Pictures\Screenshots\Screenshot 2025-10-18 214234.png"
OUT_IMG        = r"C:\Users\anye forti\Desktop\output.png"
# ————————————————————————————————————————————————————————————

image = cv2.imread(IN_IMG)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

r = image_rgb[:, :, 0].astype(np.float32)
g = image_rgb[:, :, 1].astype(np.float32)
b = image_rgb[:, :, 2].astype(np.float32)

rg_diff = np.abs(r - g)
rg_avg_minus_b = (r + g) / 2 - b
min_rg = np.minimum(r, g)

chick_mask = ((rg_diff <= 16) & 
				(rg_avg_minus_b >= 20) & 
				(min_rg >= 80) &
				(r > b) & 
				(g > b))

filtered = np.zeros_like(image_rgb)
filtered[chick_mask] = image_rgb[chick_mask]

filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR)
cv2.imwrite(OUT_IMG, filtered_bgr)
