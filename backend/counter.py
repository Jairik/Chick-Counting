'''
This file will be used for the creation of an ease of use method for loading in a counter of the desired type
'''

from typing import Literal
import joblib
import yolo_implementation
import clustering
import inspect
from data_processing import *

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
		self._using_preprocess_pipeline = False if(len(image_pipeline) == 0) else True

		#now we win define a functional pipeline for image simplification
		#the pipeline will be in class-local saved variable from initiation
		#this will be a list of dicts
		#the list represents each transformation in the pipeline
		#each item is a dict, with kv pair being function name and dict of parameters
		#each parameter dict pair will be in standard form of param_name:param_val
		self._preprocess_pipeline = image_pipeline

		#NOTE now we are going to validate the pipeline brought in END#NOTE

		if(len(image_pipeline)>0):
			
			is_legal = True

			#list collection of function names
			func_names = []

			#check each function info set
			for f_i, func in enumerate(image_pipeline):

				#provided function name should be callable
				if(not callable(str(func.key()))):
					is_legal = False
					break

				#now check each parameter set for each function
				sig = inspect.signature(func.key())

				#use try except on bind to ensure this works
				try:
					sig.bind(**func.values())
				except Exception as e:
					raise ValueError(f"Pipeline function #{f_i+1} '{str(func.key())}'. Parameters cannot bind to function.")

			#check if function loop has been broken
			if(not is_legal):
				raise ValueError(f"Pipeline function #{f_i+1} '{str(func.key())}' is not callable.")
			
		#NOTE if here is successful, pipeline is logically working on has no contents END#NOTE
		

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
		if(self._using_preprocess_pipeline):
			processed_image = self.pipeline(image)

		#all mounted counters should operate without fault, as they all contain count functionality
		self._counter.count(processed_image)

	def pipeline(
		self,
		image	:	any
	):
		'''
		### info: ###
		This function takes in a given image, and runs it through a pipeline of transformations requested.
		'''

		#for each function in the pipeline, extract function name and parameters
		for transformer_name, transformer_params in self._preprocess_pipeline:

			#get the function from the global namespace
			transformer_func = globals().get(transformer_name)

			#ensure that an illegal function name did not bypass initial test
			if(not callable(transformer_func)):
				raise KeyError(f"Function is not callable {str(transformer_name)}")

			#apply transformation function and provided parameters to image
			image = transformer_func(**transformer_params)

			#and go again

		#return image after all transforming functions have been applied.
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
	def preprocess_pipeline(self):
		return NotImplementedError(f"make function that prints out pipeline functionality all pretty.")
	
	@preprocess_pipeline.setter
	def preprocess_pipeline(self, new:list):
		self._preprocess_pipeline = new

	@property
	def using_preprocess_pipeline(self):
		return self._using_preprocess_pipeline
	
	@using_preprocess_pipeline.setter
	def using_preprocess_pipeline(self, new:bool):
		self._using_preprocess_pipeline = new

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