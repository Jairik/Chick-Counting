"""
"""
import numpy as np
from PIL import Image

def load_image_rgb(path):
	img = Image.open(path).convert("RGB")
	return np.array(img, dtype=np.uint8)

def rgb_to_hsv_np(rgb_np):
	rgb = rgb_np.astype(np.float32) / 255.0
	r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

	max_channel = np.max(rgb, axis=1)
	min_channel = np.min(rgb, axis=1)
	chroma = max_channel - min_channel

	# hue
	h = np.zeros_like(max_channel, dtype=np.float32)
	has_chroma = chroma > 1e-12
	r_is_max = (max_channel == r) & has_chroma
	g_is_max = (max_channel == g) & has_chroma
	b_is_max = (max_channel == b) & has_chroma
	h[r_is_max] = ((g[r_is_max] - b[r_is_max]) / chroma[r_is_max]) % 6.0
	h[g_is_max] = ((b[g_is_max] - r[g_is_max]) / chroma[g_is_max]) + 2.0
	h[b_is_max] = ((r[b_is_max] - g[b_is_max]) / chroma[b_is_max]) + 4.0
	h = (h / 6.0) % 1.0

	# saturation
	s = np.zeros_like(max_channel, dtype=np.float32)
	has_v = max_channel > 1e-12
	s[has_v] = chroma[has_v] / max_channel[has_v]

	# value
	v = max_channel.astype(np.float32)

	hsv = np.stack([h, s, v], axis=1)

	return hsv

def make_features(rgb_np):
	rgb = rgb_np.astype(np.float32)
	r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
	
	rgb_sum = (r + g + b) + 1e-6
	r_chroma, g_chroma, b_chroma = r / rgb_sum, g / rgb_sum, b / rgb_sum
	rg_diff, rb_diff, gb_diff = r - g, r - b, g - b

	hsv = rgb_to_hsv_np(rgb_np)
	h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

	luminance = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)

	features = np.column_stack([
		r, g, b, 
		r_chroma, g_chroma, b_chroma,
		rg_diff, rb_diff, gb_diff,
		h, s, v,
		luminance
	]).astype(np.float32)

	return features

def post_image(rgb_img, model):
	height, width, _ = rgb_img.shape
	flat_pixels = rgb_img.reshape(-1, 3)
	prob = model.predict_proba(make_features(flat_pixels))[:, 1]
	post_map = prob.reshape(height, width).astype(np.float32)

	return post_map

def make_overlay(rgb_img, post_map, thres, dim):
	assert rgb_img.shape[:2] == post_map.shape, "Image and post_map size mismatch."
	overlay = rgb_img.astype(np.float32).copy()
	mask = post_map >= float(thres)
	overlay[~mask] *= float(dim)
	
	return np.clip(overlay, 0, 255).astype(np.uint8)