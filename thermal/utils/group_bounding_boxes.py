''' Groups overlapping bounding boxes from a YOLO results object, determining overlaps based on IoU (Intersection over Union) thresholds. '''

import numpy as np
from ultralytics.engine.results import Results, Boxes
from typing import List, Tuple, Set, Any, Optional

__all__ = ["group_bounding_boxes", "merge_group_bounding_box"]


# Group and merge bounding boxes for a given index
def group_and_merge_bounding_boxes(xyxy: np.ndarray, tracker_ids: List[Any], target_tracker_id: Any, iou_thresh: float = 0.05) -> Optional[Tuple[str, Any]]:
    ''' 
    Given all boxes and their tracker IDs, group and merge boxes for a specific tracker ID. 
    Parameters:
    - xyxy: (N,4) array of bounding boxes in [x1,y1,x2,y2] format.
    - tracker_ids: List of tracker IDs corresponding to each bounding box.
    - target_tracker_id: The target tracker ID to group and merge boxes.
    - iou_thresh: IoU threshold to consider boxes as overlapping.
    '''
    # Find all detections for this tracker
    target_indices = [i for i, tid in enumerate(tracker_ids) if tid == target_tracker_id]
    if not target_indices:
        return None

    # Group *only* those boxes
    sliced_boxes = xyxy[target_indices]  # shape (M, 4)
    groups = group_bounding_boxes(sliced_boxes, iou_thresh=iou_thresh)

    # The target in the sliced array is index 0 (because target_indices[0] -> sliced_boxes[0])
    target_group = None
    for g in groups:
        if 0 in g:
            target_group = g
            break
    if target_group is None:
        return None

    # Map sliced indices back to original indices
    original_group_indices = [target_indices[i] for i in target_group]

    # Merge on the original array using original indices
    merged_box = merge_group_bounding_box(xyxy=xyxy, idxs=original_group_indices)

    # Collect original tracker ids for that group
    group_ids = [tracker_ids[i] for i in original_group_indices]

    return merged_box, group_ids


# Group bounding boxes based on IoU threshold, not considering IDs
def group_bounding_boxes(xyxy: np.ndarray, iou_thresh: float = 0.05) -> List[List[int]]:
    '''
    Group indices of overlapping boxes given an (N,4) xyxy float array.
    Parameters:
    - xyxy: (N,4) array of bounding boxes in [x1,y1,x2,y2] format.
    - iou_thresh: IoU threshold to consider boxes as overlapping.
    Returns a list of groups, each a list of indices into xyxy.
    '''
    assert xyxy.ndim == 2 and xyxy.shape[1] == 4, "xyxy must be (N,4)"
    n = len(xyxy)
    used = set()
    groups: List[List[int]] = []
    for i in range(n):
        if i in used:
            continue
        g = [i]
        for j in range(i + 1, n):
            if j in used:
                continue
            if _get_iou_xyxy(xyxy[i], xyxy[j]) >= iou_thresh:
                g.append(j)
                used.add(j)
        used.update(g)
        groups.append(g)
    return groups


# Helper for merging multiple boxes into one enclosing box
def merge_group_bounding_box(xyxy: np.ndarray, idxs: list[int]) -> tuple[int,int,int,int]:
    """ Merges multiple boxes, returning a big enclosing box (clipped to ints). """
    xs1 = xyxy[idxs, 0]; ys1 = xyxy[idxs, 1]
    xs2 = xyxy[idxs, 2]; ys2 = xyxy[idxs, 3]
    x1 = int(np.floor(xs1.min())); y1 = int(np.floor(ys1.min()))
    x2 = int(np.ceil(xs2.max()));  y2 = int(np.ceil(ys2.max()))
    return x1, y1, x2, y2


# Helper to compute IoU between two boxes
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
    
    # Return the IoU
    return inter / union