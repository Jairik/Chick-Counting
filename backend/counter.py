'''
This file will be used for the creation of an ease of use method for loading in a counter of the desired type
'''

from typing import Literal


class Counter():
	'''
	This class will load and mount a desired counter method
	'''
	def __init__(
		self,
		counter_type	:	Literal['YOLO','Clustering']	=	None,
		counter_kwargs	:	dict							=	{}
	):
		assert (counter_type != None), "Counter_type not defined for class 'Counter'. Please select a valid option."

		#define the counter type based off of counter method used
		match(counter_type):
			case 'YOLO':
				pass
			case 'Clustering':
				pass

	def count():
		'''This function will be used to execute the model on the given frame with given parameters'''
		return
	

	@property
	def counter(self):
		return self._counter

	@counter.setter
	def counter(self, new:any=None):
		self._counter = new