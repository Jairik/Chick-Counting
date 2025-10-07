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
		self
	):
		pass

	def count(
		self,
		image	:	any
	):
		
		#convert to gray
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

		#get binary thresh of the image
		ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

		#detect the countours on the binary image using cv2. chain approximate none
		contours, hierarchy = cv2.findContours(
			image=thresh, 
			mode=cv2.RETR_TREE, 
			method=cv2.CHAIN_APPROX_NONE
		)

		return cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
	

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