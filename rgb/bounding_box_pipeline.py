''' Defines functions to map bounding box cooridates to values and run helper models to determine box-specific counts 
Contents:
    - get_box_count: Main function to map bounding box coordinates to a specific chick count
    - validate_bounding_box: Helper function to ensure bounding box is valid
    - get_box_features: Helper function to engineer specific features from the bounding box data
    - get_model_prediction: Helper function to run a model on the bounding box features to estimate chick counts
'''

import joblib  # Loading models
import numpy as np  # General computations and stuff
from typing import List, Tuple  # Type hinting
from ultralytics.engine.results import Results as YOLOResults  # YOLO results object
from ultralytics.engine.results import Boxes  # YOLO Boxes object
from sklearn.svm import SVC  # Example model, replace with actual model as needed
import utils.get_bounding_box_features as gbbf  # Importing feature extraction functions
import cv2  # Image processing
from datetime import datetime  # Timestamping for saved files
import csv  # Saving results to CSV

__all__ = ["get_box_count"]

MIN_BOX_THRESHOLD = 10  # Minimum size of a bounding box to be considered valid (in pixels) - May need to be adjusted
MODEL = SVC()  # Example model type

# Main functionality to map bounding box coordinates to a specific chick count
def get_box_count(
    box: Boxes,  # A specific box from a YOLO result
    yellow_scaled_frame: np.array,  # The full numpy array of data, scaled for "amount" of yellow, of the full frame 
    model = MODEL,  # The pre-trained model to estimate chick counts
) -> int:
    '''
    Map a bounding box to a specific chick count.
    Parameters:
        box: A specific yolo box instance
        yellow_data: The full numpy array of yellow data for the entire frame
    Returns:
        int: Estimated number of chicks in the given bounding box
    '''
    
    # Extract necessary data
    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()  # Dimensions and position from the box
    
    # Ensure that the bounding box is valid before computing
    if not validate_bounding_box([x_min, y_min, x_max, y_max], yellow_scaled_frame.shape): return
    
    # Extract the specific yellow values of the bounding box from the full frame
    box_yellow_data = yellow_scaled_frame[int(y_min):int(y_max), int(x_min):int(x_max)]
    
    # Extract specific features from the bounding box data
    box_features = get_box_features(box_yellow_data)
    
    # Run the helper model to predict counts in specific bounding boxes
    estimated_count = get_model_prediction(features=box_features, model=model, save_all=True, yellow_data=box_yellow_data)
    
    return estimated_count

# Helper function to validate that a bounding box is within image boundaries and has a reasonable size
def validate_bounding_box(
    box_coords: List[int],  # Coordinates for the bounding box
    image_shape: Tuple[int, int]  # Shape of the full frame as (height, width)
) -> bool:
    '''
    Validate if a bounding box is within the image boundaries and has a reasonable size
    Parameters:
        box (List[float]): Bounding box in the format [x_min, y_min, x_max, y_max]
        image_shape (Tuple[int, int]): Shape of the image as (height, width)
    Returns:
        bool: True if the bounding box is valid, False otherwise
    '''
    height, width = image_shape
    x_min, y_min, x_max, y_max = map(int, box_coords)
    valid: bool = True
    
    # Ensure coordinates are within image boundaries
    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        valid = False

    # Ensure box has a reasonable size
    elif (x_max - x_min) < MIN_BOX_THRESHOLD or (y_max - y_min) < MIN_BOX_THRESHOLD:
        valid = False
    
    return valid

# Helper function to engineer specific features from the bounding box data
def get_box_features(
    box: np.array,  # A specific box instance from a results object
) -> np.ndarray:
    '''
    Extract features from the contents of a bounding box to pass into a model
    Parameters:
        box: A numpy array of yellow values
    Returns:
        np.ndarray: Extracted features (e.g., mean yellow, variance, etc.)
    '''
    x = list()  # Initializing list to hold features
    
    # Feature Extraction - TODO TEST THESE & OPTIMIZE
    x.append(gbbf.get_area(box))  # Total area of the bounding box
    x.append(gbbf.get_mean_yellow(box))  # Mean yellow in the bounding box
    x.append(gbbf.get_yellow_std(box))  # Standard deviation of yellows
    x.append(gbbf.get_yellow_variance(box))  # Variance of yellows
    x.append(gbbf.get_yellow_range(box))  # Range of yellows
    x.append(gbbf.get_pixels_over_threshold(box, threshold=30.0, relative=True))  # Relative count of pixels over certain threshold
    x.append(gbbf.get_pixels_under_threshold(box, threshold=30.0, relative=True))  # Relative count of pixels under certain threshold
    x.append(gbbf.get_mean_distance_from_threshold(box, threshold=30.0))  # Mean distance from 30C
    x.append(gbbf.get_aspect_ratio(box))  # Aspect ratio of the bounding box
    x.append((gbbf.get_estimated_segment_objects_scipy(box, threshold=30.0)))  # Estimated number of objects in the box using scipy labeling
    x.append((gbbf.get_estimated_segment_objects_contours(box, threshold=35.0)))  # Estimated number of objects in the box using cv2 contours
    
    # Convert to numpy array and return
    return np.array(x)
    
# Helper function to run a model on the bounding box features to estimate chick counts
def get_model_prediction(
    features: any,  # The features extracted from the bounding box data
    model: any,  # The pre-trained model to estimate chick counts
    save_all: bool = False,  # Flag to save each prediction for analysis
    yellow_data: np.array = None  # The raw yellow data for the bounding box, if saving is enabled
) -> int:
    '''
    Run a helper model on the extracted features to estimate chick counts
    Parameters:
        features: The features extracted from the bounding box data
    Returns:
        int: Estimated number of chicks in the bounding box
    '''
    results: int =  model.predict([features])[0]
    if save_all:
        save_model_results(yellow_data=yellow_data, features=features, results=results)
    return results
    
# Helper function to save the results for analysis
def save_model_results(
    yellow_data: np.ndarray,  # The raw yellow data for the bounding box
    features: np.ndarray,  # The features extracted from the bounding box data
    pred: int  # Predicted count from the model
) -> None:
    '''
    Save the model results for analysis
    '''
    # Determining filepaths for each frame and the overall features CSV
    file_save_path_img = f"rgb_model_bb_results/frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"  # Unique filepath for each frame
    file_save_path_features = f"rgb_model_bb_results/features.csv"  # Filepath for all tested features
    
    # Normalize the yellow data, add a text label, and save it
    norm_yellow_data = cv2.normalize(yellow_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.putText(norm_yellow_data, f'Predicted Count: {pred}', (10, norm_yellow_data.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1, cv2.LINE_AA)
    cv2.imwrite(file_save_path_img, norm_yellow_data)  # Save the image
    
    # Append the features and prediction to the CSV file
    with open(file_save_path_features, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(features) + [pred])  # Save features and prediction