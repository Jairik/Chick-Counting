'''
Clustering  -   Logan Kelsch 3/2/25
This file will contain all counting models that use clustering.
'''

#NOTE all clustering models are required to have identical class functionality
#contain exact parameterizing for identical function names

import cv2
from typing import Literal


class contour():
	'''
	### info: ###
	The contour method will utilize CV2's contour functionality.
	'''
	def __init__(
	):
		pass

	def count(
		image	:	any,
		color_order	:	Literal['rgb','grb','therm']
	):
		match(color_order):
			case 'rgb':

				#use cv2 built in function ot convert the RGB format to grayscale
				gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			
			case 'brg':

				#use cv2 built in function to convert the BGR format to grayscale
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			case 'therm':

				#do something
				pass

		#grayscale is now comeplete
		#now we need to binarize the image based off of pixel values

		ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

		return
	

class temporal_crf():
	'''
	NOTE DEV INFO HERE END#NOTE
	'''
	def __init__(
		dp_here
	):
		return


class kmeans():
	'''
	NOTE DEV INFO HERE END#NOTE
	'''
	def __init__(
		dev_params_here
	):
		return
	
	def count():
		'''
		NOTE DEV ensure this has any excess info generation that
		may be desired for future model examination.
		NOTE DEV then ensure that this is either enforced in code
		FOR THE DEV or ensure that there is at least specs in the file header info box.
		'''
		return

class complete_linkage():
	'''
	NOTE This has time complexity O(N^2), will push to a later option
	'''
	def __init__():
		return

	def count():
		return


class single_linkage():
	'''
	NOTE This has time complexity O(N^2), will push to a later option
	'''
	def __init__():
		return
	
	def count():
		return