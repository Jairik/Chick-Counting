''' Groups overlapping bounding boxes from a YOLO results object, determining overlaps based on IoU (Intersection over Union) thresholds. '''

import numpy as np
from ultralytics.engine.results import Results, Boxes
from typing import List, Tuple, Set

__all__ = ["group_bounding_boxes", "merge_group_bounding_box"]

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

def group_bounding_boxes(
    xyxy: np.ndarray,
    iou_thresh: float = 0.05
) -> List[List[int]]:
    """
    Group indices of overlapping boxes given an (N,4) xyxy float array.
    Returns a list of groups, each a list of indices into xyxy.
    """
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