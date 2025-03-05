'''
This file will be used for the creation of an ease of use method for loading in a counter of the desired type
'''

from typing import Literal
import joblib
import yolo_implementation
import clustering

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
	'''
	def __init__(
		self,
		counter_type	:	Literal['YOLO','Clustering']	=	None,
		counter_kwargs	:	dict							=	{}
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
		self._detected_centers = []

		#this variable is NOTE POSSIBLY TEMPORARY NOTE for tracking projection of centers
		#this variable will be a tuple of projected X, Y offset for next frame
		self._detected_projections = []

		#this variable will be a dictionary (str:int) of detection classifications and totals over time
		self._detected_totals = {}

	#count function, is called on each frame
	def count(
		self,
		image		:	any	=	None
	):
		'''This function will be used to execute the model on the given frame with given parameters'''

		#all mounted counters should operate without fault, as they all contain count functionality
		self._counter.count(image)

		
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