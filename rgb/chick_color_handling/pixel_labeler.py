"""
This script loads an interactive pixel labeling tool for creating training data on chick/non-chick colored pixels.

Inputs:
 - IMG_DIR_PATH           -> folder containing images
 - IMG_EXT                -> file extension for program to parse
 - MAX_PIXELS             -> maximum num of pixels to label for datasets

 Outputs:
 - CHICK_PIXELS_PATH      -> npy file containing rgb values found in chick
 - NON_CHICK_PIXELS_PATH  -> npy file containing rgb values not found in chick
 - PIXEL_INDEX_PATH       -> pkl file containing list of shuffled pixels to label

Existing output paths will be appended when reran. Further instructions included while running program.
"""

import pickle
import cv2
import numpy as np
from pathlib import Path
import sys

# ——— CONFIG —————————————————————————————————————————————————
IMG_DIR_PATH             = ""
CHICK_PIXELS_PATH        = ""
NON_CHICK_PIXELS_PATH    = ""
PIXEL_INDEX_PATH         = ""
IMG_EXT                  = ".png"
MAX_PIXELS               = 100000
# ————————————————————————————————————————————————————————————

class PixelLabeler:
	def __init__(self, img_dir, chick_path, non_chick_path, index_path, img_ext):
		self.img_dir = Path(img_dir)
		self.chick_path = chick_path
		self.non_chick_path = non_chick_path
		self.index_path = index_path
		self.img_ext = img_ext

		self.chick_pixels = self.load_pixels(chick_path)
		self.non_chick_pixels = self.load_pixels(non_chick_path)

		self.image_paths = sorted(list(self.img_dir.glob(f"*{img_ext}")))
		if not self.image_paths:
			print(f"No images found in {img_dir}")
			sys.exit(1)

		self.pixel_queue, self.current_index = self.load_index()

		self.current_image = None
		self.current_pixel_pos = None
		self.current_pixel_rgb = None

	def load_pixels(self, path):
		try:
			data = np.load(path)
			print(f"Loaded {len(data)} pixels from {path}")
			return list(data)
		except FileNotFoundError:
			print(f"No existing data at {path}, starting fresh")
			return []
		
	def load_index(self):
		try:
			with open(self.index_path, 'rb') as f:
				data = pickle.load(f)
				pixel_queue = data['pixel_queue']
				current_index = data['current_index']
				print(f"Loaded pixel index: {current_index}/{len(pixel_queue)} pixels labeled")
				return pixel_queue, current_index
		except FileNotFoundError:
			print("Creating new pixel index ...")

			image_info = []
			total_pixels = 0

			for img_idx, img_path in enumerate(self.image_paths):
				img = cv2.imread(str(img_path))
				if img is None:
					continue
				h, w = img.shape[:2]
				pixels_in_image = h * w
				image_info.append({
					'img_idx': img_idx,
					'width': w,
					'pixel_count': pixels_in_image,
					'cumulative_start': total_pixels
				})
				total_pixels += pixels_in_image

			print(f"Total pixels across all images: {total_pixels:,}")

			num_to_sample = min(MAX_PIXELS, total_pixels)
			print(f"Randomly sampling {num_to_sample:,} pixels ...")

			random_indices = np.random.choice(total_pixels, size=num_to_sample, replace=False)

			pixel_queue = []
			for global_idx in random_indices:
				for img_info in image_info:
					if global_idx < img_info['cumulative_start'] + img_info['pixel_count']:
						local_idx = global_idx - img_info['cumulative_start']
						y = local_idx // img_info['width']
						x = local_idx % img_info['width']
						pixel_queue.append((img_info['img_idx'], y, x))
						break

			print(f"Indexed {len(pixel_queue):,} random pixels from {len(self.image_paths)} images")
			return pixel_queue, 0
				
	def save_and_exit(self):
		print("\n\nSaving data ...")

		if self.chick_pixels:
			np.save(self.chick_path, np.array(self.chick_pixels, dtype=np.uint8))
			print(f"Saved {len(self.chick_pixels)} chick pixels")

		if self.non_chick_pixels:
			np.save(self.non_chick_path, np.array(self.non_chick_pixels, dtype=np.uint8))
			print(f"Saved {len(self.non_chick_pixels)} non-chick pixels")

		with open(self.index_path, 'wb') as f:
			pickle.dump({
				'pixel_queue': self.pixel_queue,
				'current_index': self.current_index
			}, f)
		print(f"Saved progress: {self.current_index}/{len(self.pixel_queue)} pixels labeled")

		cv2.destroyAllWindows()
		sys.exit(0)

	def get_next_pixel(self):
		if self.current_index >= len(self.pixel_queue):
			print("\nAll pixels labeled")
			self.save_and_exit()

		img_idx, y, x = self.pixel_queue[self.current_index]

		if self.current_image is None or img_idx != getattr(self, 'current_img_idx', -1):
			img_path = self.image_paths[img_idx]
			self.current_image = cv2.imread(str(img_path))
			self.current_img_idx = img_idx

		self.current_pixel_pos = (x, y)

		b, g, r = self.current_image[y, x]
		self.current_pixel_rgb = (int(r), int(g), int(b))

	def display_image(self):
		display = self.current_image.copy()
		x, y = self.current_pixel_pos
		h, w = display.shape[:2]

		if x < w // 2 and y < h // 2:
			start = (x+20, y+20)
			end = (x+1, y+1)
		elif x >= w // 2 and y < h // 2:
			start = (x-20, y+20)
			end = (x-1, y+1)
		elif x < w // 2 and y >= h // 2:
			start = (x+20, y-20)
			end = (x+1, y-1)
		else:
			start = (x-20, y-20)
			end = (x-1, y-1)

		cv2.arrowedLine(display, start, end, (0, 0, 255), 2, tipLength=0.3)
		cv2.imshow("Image", display)

		info_panel = np.zeros((200, 600, 3), dtype=np.uint8)
		
		r, g, b = self.current_pixel_rgb
		text = f"RGB: ({r}, {g}, {b})"
		cv2.putText(info_panel, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
		
		progress = f"Progress: {self.current_index}/{len(self.pixel_queue)} ({self.current_index/len(self.pixel_queue)*100:.1f}%)"
		cv2.putText(info_panel, progress, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		
		stats = f"Chick: {len(self.chick_pixels)} | Non-chick: {len(self.non_chick_pixels)}"
		cv2.putText(info_panel, stats, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		
		controls = "Press: y=chick | n=non-chick | s=skip | ESC=save & exit"
		cv2.putText(info_panel, controls, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
		
		cv2.imshow("Info", info_panel)

	def label_chick(self):
		self.chick_pixels.append(self.current_pixel_rgb)
		self.current_index += 1
		self.get_next_pixel()

	def label_non_chick(self):
		self.non_chick_pixels.append(self.current_pixel_rgb)
		self.current_index += 1
		self.get_next_pixel()

	def skip_pixel(self):
		self.current_index += 1
		self.get_next_pixel()

	def run(self):
		self.get_next_pixel()

		while True:
			self.display_image()
			key = cv2.waitKey(0) & 0xFF

			if key == ord('y'):
				self.label_chick()
			elif key == ord('n'):
				self.label_non_chick()
			elif key == ord('s'):
				self.skip_pixel()
			elif key == 27:
				self.save_and_exit()

def main():
	if not IMG_DIR_PATH:
		print("ERROR: Please set IMG_DIR_PATH in the config section")
		return
	
	labeler = PixelLabeler(IMG_DIR_PATH, CHICK_PIXELS_PATH, 
						NON_CHICK_PIXELS_PATH, PIXEL_INDEX_PATH, IMG_EXT)
	labeler.run()

if __name__ == "__main__":
	main()