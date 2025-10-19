''' Groups overlapping bounding boxes from a YOLO results object, determining overlaps based on IoU (Intersection over Union) thresholds. '''

import numpy as np
from ultralytics.engine.results import Results, Boxes
from typing import List, Tuple, Set

__all__ = ["group_bounding_boxes"]

def _get_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU for two [x1,y1,x2,y2] boxes (numpy arrays)."""
    # Intersection
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    # Union
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0

def group_bounding_boxes(
    results: Results,
    overlapping_threshold: float = 0.05
) -> List[Boxes]:
    '''
    Groups overlapping bounding boxes from a YOLO results object.
    Parameters:
        results: The YOLO results object containing bounding boxes
        overlapping_threshold: The IoU threshold to consider boxes as overlapping
    Returns:
        List[Boxes]: A list of grouped bounding boxes
    '''
    
    # Initialize list to hold and track grouped boxes
    grouped_boxes: List[Boxes] = []
    boxes: Boxes = results.boxes  # Extract boxes from results
    used_indices: Set[int] = set()  # Track indices of boxes that have been grouped
    num_boxes: int = len(boxes)
    
    # Loop through each box and group overlapping ones
    for i in range(num_boxes):
        
        # Skip already grouped boxes
        if i in used_indices:
            continue
        
        # Get current box and start a new group
        current_box: np.array = boxes.xyxy[i].cpu().numpy()  # Current box coordinates
        group: List[int] = [i]  # Start a new group with the current box
        
        # Check for overlapping boxes
        for j in range(i + 1, num_boxes):
            if j in used_indices:
                continue  # Skip already grouped boxes
            
            compare_box: np.array = boxes.xyxy[j].cpu().numpy()  # Box to compare with
            iou = _get_iou_xyxy(current_box, compare_box)  # Compute IoU
            
            if iou >= overlapping_threshold:
                group.append(j)  # Add to group if overlapping
                used_indices.add(j)  # Mark as used
        
        # Create a new Boxes object for the grouped boxes
        grouped_box_coords: np.array = np.array([boxes.xyxy[k].cpu().numpy() for k in group])
        grouped_boxes.append(Boxes(xyxy=grouped_box_coords))
        used_indices.add(i)  # Mark current box as used
        
    return grouped_boxes  # Return the list of grouped bounding boxes