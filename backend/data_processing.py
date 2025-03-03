'''
Logan Kelsch - 2/19/25 - data processing file
This file will be used for loading data, 
calling reconstruction/construction functions from feature_usage.py NOTE WHILE loading in data,
saving of constructed, augmented, modulated, or altered data for ease of collection and usage and minimization of speed-matter in 
		  data loading of training phase.
'''


def remove_some_RGB(
	image	:	any			=	None,
	red		:	float|int	=	0,
    green	:	float|int	=	0,
    blue	:	float|int	=	0
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
	assert image, "the function remove_some_RGB was called on image of type NONETYPE."


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