'''
Logan Kelsch - 2/19/25 - data processing file
This file will be used for loading data, 
calling reconstruction/construction functions from feature_usage.py NOTE WHILE loading in data,
saving of constructed, augmented, modulated, or altered data for ease of collection and usage and minimization of speed-matter in 
		  data loading of training phase.
'''

'''
	WHITEBOARD COLORS

			R		G		B
BELT		<100	<180	<230
EGGSHELL	>230	>240	>240

CHICKS		>200	>200	>150
			152		140		92
			110		93		37
			
'''

import numpy as np
import cv2


import cv2

def rgb_to_bgr(frame:any=None):
	"""
	Convert an RGB image to BGR color space.
	
	Args:
		frame (np.ndarray): Input image in RGB order.
	Returns:
		np.ndarray: Image in BGR order.
	"""
	return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(frame:any=None):
	"""
	Convert a BGR image to RGB color space.
	
	Args:
		frame (np.ndarray): Input image in BGR order.
	Returns:
		np.ndarray: Image in RGB order.
	"""
	return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def bgr_to_gray(frame:any=None):
	"""
	Convert a BGR image to a single-channel grayscale image.
	
	Args:
		frame (np.ndarray): Input image in BGR order.
	Returns:
		np.ndarray: Grayscale image.
	"""
	return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def background_segment(frame, fgbg):
	  
	'''
	this is a work-around to force 3 channels to work properly
	as a partial through MOG method of background segmentation
	'''

	mask = fgbg.apply(frame)  
	return cv2.bitwise_and(frame, frame, mask=mask)

def rgb_to_gray(frame:any=None):
	"""
	Convert a BGR image to a single-channel grayscale image.
	
	Args:
		frame (np.ndarray): Input image in BGR order.
	Returns:
		np.ndarray: Grayscale image.
	"""
	return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def gray_threshold(
	frame:any=None,
	thresh=127, maxval=255, method=cv2.THRESH_BINARY):
	"""
	Convert a BGR image to grayscale and then apply a binary threshold.
	
	Args:
		frame (np.ndarray): Input image in BGR order.
		thresh (int): Threshold value.
		maxval (int): Value to set for pixels above threshold.
		method (int): OpenCV thresholding type (e.g., cv2.THRESH_BINARY).
	Returns:
		np.ndarray: Binary (thresholded) image.
	"""
	gray = bgr_to_gray(frame)
	_, binary = cv2.threshold(gray, thresh, maxval, method)
	return binary

def mask_brightness_thresh(
	frame: any = None,
	brightness_threshold: int = 200
):
	"""
	Keep only those pixels where any color channel is above brightness_threshold.
	All other pixels are set to black.

	Args:
		frame (np.ndarray): Input image in BGR (or RGB) order.
		brightness_threshold (int): Threshold for masking bright pixels.

	Returns:
		np.ndarray: Masked image.
	"""
	# Copy to avoid mutating the original frame
	masked = frame.copy()

	# Build a 2D mask: True wherever any of the three channels > threshold
	bright_mask = np.any(masked > brightness_threshold, axis=2)

	# Expand it back to 3 channels
	full_mask = np.stack([bright_mask] * 3, axis=-1)

	# Zero‑out every pixel where full_mask is False
	masked[~full_mask] = 0

	return masked

import numpy as np

def mask_inv_rgb_window(
	frame: any = None,
	lower_threshold: int = 100,
	upper_threshold: int = 150
):
	"""
	Keep only those pixels whose R, G, and B channels all lie within
	[lower_threshold, upper_threshold]. All other pixels are set to black.

	Args:
		frame (np.ndarray): Input image in BGR or RGB order (HxWx3).
		lower_threshold (int): Lower bound of the window (inclusive).
		upper_threshold (int): Upper bound of the window (inclusive).

	Returns:
		np.ndarray: Masked image where only the “in-window” pixels remain.
	"""
	# avoid modifying the original
	masked = frame.copy()

	# mask where all three channels are within [lower, upper]
	window_mask = np.all(
		(masked >= lower_threshold) & (masked <= upper_threshold),
		axis=2
	)

	# expand to 3 channels and zero out everything else
	full_mask = np.stack([window_mask] * 3, axis=-1)
	masked[~full_mask] = 0

	return masked

import numpy as np

def mask_rgb_window(
	frame: any = None,
	lower_threshold: int = 100,
	upper_threshold: int = 150
):
	"""
	Turn all pixels whose R, G, and B channels all lie within
	[lower_threshold, upper_threshold] to black, and keep every other pixel unchanged.

	Args:
		frame (np.ndarray): Input image in BGR or RGB order (HxWx3).
		lower_threshold (int): Lower bound of the window (inclusive).
		upper_threshold (int): Upper bound of the window (inclusive).

	Returns:
		np.ndarray: Image where “in-window” pixels are zeroed out.
	"""
	# Copy to avoid mutating the original frame
	masked = frame.copy()

	# Build a mask where all three channels are within [lower, upper]
	window_mask = np.all(
		(masked >= lower_threshold) & (masked <= upper_threshold),
		axis=2
	)

	# Zero‑out pixels inside the window
	masked[window_mask] = 0

	return masked


def mask_red_thresh(
	frame	:	any	=	None,
	threshold	:	int	=	127
):
	# Copy the frame to avoid modifying original
	masked = frame.copy()

	# Red channel is channel 2 in BGR
	red_channel = masked[:, :, 2]

	# Create a mask where red > threshold
	red_mask = red_channel > threshold

	# Expand mask to 3 channels (for BGR)
	full_mask = np.stack([red_mask]*3, axis=-1)

	# Set all pixels where red <= threshold to black
	masked[~full_mask] = 0

	return masked

import numpy as np

def mask_green_thresh(
	frame: any = None,
	threshold: int = 127
):
	# Copy the frame to avoid modifying original
	masked = frame.copy()

	# Green channel is channel 1 in BGR
	green_channel = masked[:, :, 1]

	# Create a mask where green > threshold
	green_mask = green_channel > threshold

	# Expand mask to 3 channels (for BGR)
	full_mask = np.stack([green_mask] * 3, axis=-1)

	# Set all pixels where green <= threshold to black
	masked[~full_mask] = 0

	return masked


def mask_blue_thresh(
	frame: any = None,
	threshold: int = 127
):
	# Copy the frame to avoid modifying original
	masked = frame.copy()

	# Blue channel is channel 0 in BGR
	blue_channel = masked[:, :, 0]

	# Create a mask where blue > threshold
	blue_mask = blue_channel > threshold

	# Expand mask to 3 channels (for BGR)
	full_mask = np.stack([blue_mask] * 3, axis=-1)

	# Set all pixels where blue <= threshold to black
	masked[~full_mask] = 0

	return masked


def filter_frame(
	image	:	any			=	None
):
	'''
	### info: ###
	This function is the image redundancy removal pipeline
	'''

	# NOTE EXAMPLE CODE END#NOTE _____________________________________
	image = downscale(image)
	image = remove_some_RGB(image)
	image = group_some_RGB(image)
	# NOTE END EXAMPLE CODE END#NOTE _________________________________
	return image


def remove_some_RGB(
	image	:	any			=	None,
	red		:	tuple		=	0,
	green	:	tuple		=	0,
	blue	:	tuple		=	0
):
	'''
	#### NOTE TEMP dev notes: ####
	This function will either call a function from some library if it is faster, <br>
	or it will do it on its own. <br>
	### info: ###
	This function takes in an image and turns all pixels below some provided R,G,B threshold to black.
	### params: ###
	-	image:
	-	-	this is the image file coming in
	-	red/green/blue:
	-	-	These numbers can be interpreted by the function as a percent (0.0 to 1.0) or value (0 to 255).
	### returns: ###
	This function will return the image with removed pixels
	'''
	#assert image, "the function remove_some_RGB was called on image of type NONETYPE."

	if(type(red) == float):
		red *= 255
	if(type(green) == float):
		green *= 255
	if(type(blue) == float):
		blue *= 255

	#make a mask for the desired colors
	if(red != 0):
		mask = image[:, :, 2] >= red[1]
		mask = image[:, :, 2] <= red[0]
	if(green != 0):
		mask = image[:, :, 1] >= green
	if(blue != 0):
		mask = image[:, :, 0] >= blue

	filtered_image = np.zeros_like(image)
	filtered_image[mask] = image[mask]

	#cv2.imwrite('new_img.jpg', filtered_image)
	#cv2.imshow('f img', filtered_image)
	#cv2.waitkey(0)
	#cv2.destroyAllWindows()

	return filtered_image


def group_some_RGB(
	image	:	any		=	None,
	red		:	tuple	=	None,
	green	:	tuple	=	None,
	blue	:	tuple	=	None,
	group_color	:	tuple	=	(1, 1, 1)
):
	'''
	### info: ###
	This function will take given pixel color ranges and turn them into a specified color.
	### params: ###
	-	image:
	-	-	this is the image file coming in
	-	red,green,blue:
	-	-	These are interpreted as tuples of a lower bound and upper bound, and are optional for each color.
	-	-	These bounds can be comprised of pairs of percents ex:(0.1 , 0.3) or as values ex:(55 , 155).
	### returns: ###
	This function will return the image with altered coloring of those ranges
	'''
	assert image, "the function remove_some_RGB was called on image of type NONETYPE."


def downscale(
	image	:	any			=	None,
	down	:	int|tuple	=	1
):
	'''
	### info: ###
	This function will return a downscaled image of a provided image.
	### params: ###
	-	image:
	-	-	this is the image file coming in
	-	down:
	-	-	this is the downscaling parameter, which can either be a downscaling factor, or a resolution
	-	-	-	resolution - interpretable as a tuple
	-	-	-	ds-factor  - interpretable as an integer
	### returns: ###
	this function will return the downscaled version of the image
	'''
	assert image, "the function remove_some_RGB was called on image of type NONETYPE."