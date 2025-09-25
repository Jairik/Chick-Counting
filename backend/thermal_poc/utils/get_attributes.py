''' Utility functions to extract specific data from a YOLO model prediction '''

__all__ = ["get_bounding_boxes", "get_probs", "get_keypoints"]

def get_bounding_boxes(results):
    '''
    Return the bounding boxes from a model's results
    Parameter: results - The result of a YOLO model's image prediction
    Returns: An array of bounding boxes 
    '''
    boxes: list = []
    # Loop through each prediction made from the model
    for result in results:
        boxes.append(result.boxes)  # Bounding box outputs
    return boxes

def get_probs(results):
    '''
    Return the probabilities for each prediction from a model's results
    Parameter: results - The result of a YOLO model's image prediction
    Returns: An array of probability objects (probs)
    '''
    probs: list = []
    # Loop through each prediction made from the model
    for result in results:
        probs.append(result.probs)
    return probs


def get_keypoints(results):
    '''
    Return the keypoints of predictions from a model's results
    Parameter: results - The result of a YOLO model's image prediction
    Returns: An array of keypoints (areas of interest for the model)
    Use: Could be beneficial when designing contour models
    '''
    keypoints: list = []
    for result in results:
        keypoints.append(result.keypoints)
    return keypoints
