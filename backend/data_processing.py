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


def mask_red_thresh(frame, red_threshold=127):
    # Copy the frame to avoid modifying original
    masked = frame.copy()

    # Red channel is channel 2 in BGR
    red_channel = masked[:, :, 2]

    # Create a mask where red > threshold
    red_mask = red_channel > red_threshold

    # Expand mask to 3 channels (for BGR)
    full_mask = np.stack([red_mask]*3, axis=-1)

    # Set all pixels where red <= threshold to black
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