"""
This script extracts rgb values from every pixel in all images in a folder and saves them as one NumPy array of uint8.
"""
from pathlib import Path
from PIL import Image
import numpy as np

# ——— CONFIG —————————————————————————————————————————————————
IMG_DIR_PATH   = ""
OUTPUT_PATH    = ""
IMG_EXT        = ".png"
# ————————————————————————————————————————————————————————————

def main():
	path = Path(IMG_DIR_PATH)
	blocks = []

	for img_path in path.glob(f"*{IMG_EXT}"):
		with Image.open(img_path) as im:
			a = np.array(im.convert("RGB"), dtype=np.uint8)
		blocks.append(a.reshape(-1, 3))

	if not blocks:
		print("No images found.")
		return
	
	rgb = np.vstack(blocks)
	np.save(OUTPUT_PATH, rgb)

	print(f"Saved {rgb.shape} to {OUTPUT_PATH}")

if __name__ == "__main__":
	main()