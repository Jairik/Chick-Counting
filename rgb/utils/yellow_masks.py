''' Helper functions for generating yellow mask (numpy array) of full image, prior to bounding box processing '''

import cv2
import numpy as np
from typing import Tuple

__all__ = ['get_yellow_mask']

def get_yellow_mask(image: np.array, normalize: bool = true) -> np.array:
    '''
    Generate a yellow mask of the image
    Parameters:
        image: A numpy array of the image
        normalized: Whether to normalize the mask to [0, 1] range
    Returns:
        np.array: A numpy array of the yellow mask
    '''
    
    # Ensure the image if of float32 representation
    img = img.astype(np.float32) / 255.0 if normalized else img.astype(np.float32)
    
    # Extract RGB values
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]
    
    # Measuring yellow intensity of each pixel. We are looking for:
    #  - High R and G values
    #  - Low B values
    #  - Small difference between R and G values
    yellow_mask = (R + G) / 2 - (1 - B) * (np.abs(R - G))
    
    # If normalize is set, normalize the mask to [0, 1] range
    if normalize:
        yellow_mask = np.clip(yellow_mask, 0, 1)
        
    return yellow_mask