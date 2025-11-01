''' Miscellanious utility functions for thermal image feature extraction '''

import numpy as np

def get_area(box: np.array) -> int:
    '''
    Get the area of the bounding box
    '''
    return box.shape[0] * box.shape[1]

def get_mean_temperature(box: np.array) -> float:
    '''
    Get the mean temperature of the bounding box
    '''
    return np.mean(box)

def get_pixels_over_threshold(box: np.array, threshold: float = .9, relative: bool = True) -> int:
    '''
    Get the number of pixels over a certain temperature threshold
    '''
    x: float = np.sum(box > threshold)
    if(relative):
        return x / get_area(box)
    return x

def get_pixels_under_threshold(box: np.array, threshold: float = .9, relative: bool = True) -> int:
    '''
    Get the number of pixels under a certain temperature threshold
    '''
    x: float = np.sum(box < threshold)
    if(relative):
        return x / get_area(box)
    return x
    
def get_temperature_range(box: np.array) -> float:
    '''
    Get the temperature range of the bounding box
    '''
    return np.max(box) - np.min(box)

def get_temperature_std(box: np.array) -> float:
    '''
    Get the standard deviation of the temperatures in the bounding box
    '''
    return np.std(box)

def get_temperature_variance(box: np.array) -> float:
    '''
    Get the variance of the temperatures in the bounding box
    '''
    return np.var(box)

def get_mean_distance_from_threshold(box: np.array, threshold: float = .9) -> float:
    '''
    Get the mean distance from a certain temperature threshold
    '''
    return np.mean(np.abs(box - threshold))

def get_aspect_ratio(box: np.array) -> float:
    '''
    Get the aspect ratio of the bounding box
    '''
    return box.shape[0] / box.shape[1]

def get_estimated_segment_objects_scipy(box: np.array, threshold: float = .9) -> int:
    '''
    Estimate the number of objects in a bounding box by segmenting based on a temperature threshold
    '''
    from scipy.ndimage import label
    mask = (box > threshold).astype(np.unit8)
    labeled, num_objects = label(mask)
    return num_objects

def get_estimated_segment_objects_contours(box: np.array, threshold: float = .9) -> int:
    '''
    Estimate the number of objects in a bounding box by segmenting based on a temperature threshold using contours
    '''
    import cv2
    _, binary = (cv2.threshold(box.astype(np.float32), threshold, 255, cv2.THRESH_BINARY)).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)