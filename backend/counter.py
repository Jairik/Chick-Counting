'''
This file will be used for the creation of an ease of use method for loading in a counter of the desired type
'''

from typing import Literal
import joblib
import yolo_implementation
import clustering
import inspect
from data_processing import *
import inspect
from functools import partial
import cv2

#counter class, acts as a mount for yolo or custom models
class Counter():
	'''
	### info: ###
	This class will load and mount a desired counter method.
	### params: ###
	-	counter-type:
	-	-	Type of clustering model that will be used.
	-	counter-kwargs:
	-	-	Dict variable containing all parameters and values for the counter.
	-	image_pipeline:
	-	-	A collection of instructions on how to preprocess the data before counting.

	'''
	def __init__(
		self,
		counter_type	:	Literal['YOLO','Clustering']	=	None,
		counter_kwargs	:	dict							=	{},
		img_norm_mode	:	Literal['trans_list','bg_seg','None']	=	'trans_list',
		image_pipeline	:	list							=	[]
	):
		assert (counter_type != None), "Counter_type not defined for class 'Counter'. Please select a valid option."

		#define the counter type based off of counter method used
		match(counter_type):

			#if we are working with a YOLO based counter, make this class act as a mount
			case 'YOLO':
				self._counter = yolo_implementation.YOLO_ObjectCounter(**counter_kwargs)

			#if we are working with a custom clustering model, NOTE DEV HERE END#NOTE
			case 'Clustering':
				raise NotImplementedError(f'Counter type {counter_type} has not yet been implemented.')
			
		#this variable keeps the string variable of the type of counter, ex: YOLO
		self._counter_type = counter_type

		#this variable is a list of tuples, with each tuple being a coordinate of a center
		self._detected_object_centers = []

		#this variable is NOTE POSSIBLY TEMPORARY NOTE for tracking projection of centers
		#this variable will be a tuple of projected X, Y offset for next frame
		self._detected_object_vectors = []

		#this variable will be a dictionary (str:int) of detection classifications and totals over time
		self._detected_totals = {}

		#collect boolean info based off of pipeline brought in
		self._img_norm_pipeline_mode = img_norm_mode

		#now we win define a functional pipeline for image simplification
		#the pipeline will be in class-local saved variable from initiation
		#this will be a list of dicts
		#the list represents each transformation in the pipeline
		#each item is a dict, with kv pair being function name and dict of parameters
		#each parameter dict pair will be in standard form of param_name:param_val
		#NOTE here we validate the pipeline brought in and assign the call list of inormfuncs END#NOTE
		self._img_norm_pipeline = self.build_pipeline(image_pipeline)

		#NOTE if here is successful, pipeline is logically working OR has no contents END#NOTE
		

	#count function, is called on each frame
	def count(
		self,
		image		:	any	=	None
	):
		'''This function will be used to execute the model on the given frame with given parameters'''

		'''NOTE BEGIN DEV NOTE BEGIN DEV NOTE
		Consider the use of parallelism in image pipeline processing.
		the RBPi will have 4 cores, and a Logan suggested library for this 
		computational parallelism would be
		import concurrent.futures 
		make numpy array of images
		with concurrent.futures.ThreadPoolExecutor() as Executor:
			processed_images = list(executor.map(process_img_func, images))
		also consider use of cv2.cuda library
		NOTE END DEV NOTE END DEV NOTE'''

		#until then... (regarding dev note above)
		#run the provided image through a processing pipeline per user request
		if(self._img_norm_pipeline_mode != 'None'):
			processed_image = self.pipeline(image)

		#all mounted counters should operate without fault, as they all contain count functionality
		return self._counter.count(processed_image)



	def build_pipeline(
		self,
		image_pipeline
	):
		"""
		### info: ###
		This function considers the pipeline mode that is going to be used in the hot loop for detection, and builds it accordingly.
		### params: ###
		- image-pipeline:
		- - this is a list of desired transformations to be make to each frame. (OPTIONAL) EXAMPLE ON BOTTOM OF THIS WINDOW.
		- - if the mode is set to bg_seg, then transformations are irrelevant as color
		  - channels are destroyed, and this variable will not be considered.
		EXAMPLE: 
		>>> image_pipeline = [
		>>> {'mask_red_thresh':   {'red_threshold': 127}},
		>>> {'normalize_sigmoid': {'gain': 1.2, 'cutoff': 0.5}}
		>>> ]
		"""

		#a call list will be used regardless of approach for simplicity
		call_list = []

		#check which mode the pipeline is set to for proper construction
		match(self._img_norm_pipeline_mode):

			#using a transformation list, instead of cv2 background segmentation method
			case 'trans_list':

				#for each function provided in desired format
				for idx, func_map in enumerate(image_pipeline):

					# unpack the one‐item dict
					func_name, params = next(iter(func_map.items()))

					# resolve and validate
					fn = globals().get(func_name)
					if not callable(fn):
						raise ValueError(f"#{idx}: '{func_name}' is not defined or not callable")

					#collect signature of function
					sig = inspect.signature(fn)

					#attempt binding the parameters to the signature
					try:
						sig.bind_partial(**params)
					except TypeError as e:
						raise ValueError(f"#{idx}: bad params {params!r} for {func_name}{sig}\n -> {e}")

					# bind parameters into a partial so our hot loop is simpler
					call_list.append(partial(fn, **params))

			#using cv2's background MOG method segmentation function instead of transformation list
			case 'bg_seg':

				#load in segmentation model
				fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

				#generate a partial using a preprocessing function created in the data_preprocessing file
				#this function is simply for routing 'frame' being passed in the hot loop properly to the function
				#all this does is call fgbg.apply( on frame ) and then reconstructs the
				#3 color channels so that the YOLO model is being passed a 3d ndarray with the dimensions it was expecting.
				call_list.append(partial(background_segment, fgbg=fgbg))
			
			case 'None':
				pass #nothing needs appended to call list

		#under all circumstances, call list is complete and ready 
		#to be used a normalization stepper in hot loop
		return call_list


	def pipeline(
		self,
		image	:	any
	):
		'''
		### info: ###
		This function takes in a given image, and runs it through a pipeline of transformations requested.
		'''

		for step in self._img_norm_pipeline:
			image = step(image)    # step is already a fn(img, **params)

		#return image after all transforming functions have been applied
		return image

		
	#save model for storing a preferred counter
	def save(
		folder_save_name	:	str	=	None
	):
		'''
		### info: ###
		This function will save the given counter to a specified folder. <br>
		joblib is used for saving and loading.
		#### The name parameter should have no extention, as it will be a parent folder's name.
		'''

		assert folder_save_name, "Tried to save the counter, but no counter name was specified"

	#load model for loading a preferred counter
	def load(
		folder_load_name	:	str	=	None
	):
		'''
		### info: ###
		This function will load a prameter from the given parent folder name. <br>
		joblib is used for saving and loading.
		#### the name parameter should have no extention, as it will be a parent folder's name.
		'''
		
	###					Here is the end of class function and operational development.					 ###
	### NOTE NOTE property and setter class defintions will be placed below this line. END#NOTE END#NOTE ###
	### ________________________________________________________________________________________________ ###

	#pipeline variables
	@property
	def img_norm_pipeline(self):
		return NotImplementedError(f"make function that prints out pipeline functionality all pretty.")
	
	@img_norm_pipeline.setter
	def img_norm_pipeline(self, new:list):
		self._img_norm_pipeline = new

	@property
	def img_norm_pipeline_mode(self):
		return self._img_norm_pipeline_mode
	
	@img_norm_pipeline_mode.setter
	def img_norm_pipeline_mode(self, new:bool):
		self._img_norm_pipeline_mode = new

	#detected centers

	@property
	def detected_centers(self):
		return self._detected_centers

	@detected_centers.setter
	def detected_centers(self, new:any=None):
		self._detected_centers = new

	#detected projections

	@property
	def detected_projections(self):
		return self._detected_projections

	@detected_projections.setter
	def detected_projections(self, new:any=None):
		self._detected_projections = new

	#detected totals

	@property
	def detected_totals(self):
		return self._detected_totals

	@detected_totals.setter
	def detected_totals(self, new:any=None):
		self._detected_totals = new

	#counter type

	@property
	def counter_type(self):
		return self._counter_type
	
	@counter_type.setter
	def counter(self, new:any=None):
		self._counter_type = new

	#counter, mounted

	@property
	def counter(self):
		return self._counter

	@counter.setter
	def counter(self, new:any=None):
		self._counter = new