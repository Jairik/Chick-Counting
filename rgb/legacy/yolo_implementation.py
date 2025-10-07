'''
YOLO implementation - Logan Kelsch 3/2/25
This file will be for the isolation of YOLO libraries on the .py file backend.
'''

import ultralytics
import ultralytics.solutions

def YOLO_ObjectCounter(*args, **kwargs):
    '''
    this function grabs the Object Counter from solutions in ultralytics
    '''
    return ultralytics.solutions.ObjectCounter(
        *args,
        **kwargs
    )