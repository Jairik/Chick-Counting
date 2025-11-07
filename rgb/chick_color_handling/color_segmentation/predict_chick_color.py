"""
"""
# ——— CONFIG —————————————————————————————————————————————————
MODEL_PATH         = ""
IMAGE_PATH         = ""
OUTPUT_PATH        = ""
THRESHOLD          = 0.90
DIM_FACTOR         = 0.00
# ————————————————————————————————————————————————————————————

import os, joblib
from PIL import Image
from chick_color_lib import load_image_rgb, post_image, make_overlay

def save_image_rgb(path, overlay_np):
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	Image.fromarray(overlay_np, mode="RGB").save(path)

def main():
	assert os.path.exists(MODEL_PATH), "missing model"
	assert os.path.exists(IMAGE_PATH), "missing image"

	model = joblib.load(MODEL_PATH)
	img = load_image_rgb(IMAGE_PATH)

	post = post_image(img, model)
	overlay = make_overlay(img, post, thres=THRESHOLD, dim=DIM_FACTOR)

	save_image_rgb(OUTPUT_PATH, overlay)

if __name__ == "__main__":
	main()