# Bounding Box Pipeline

This pipeline aims at validating the contents of each bounding box, running a meta-model (on top of YOLO predictions) to obtain a more accurate per-box count. Note that this does not account for overlapping frames, so this should be an additional consideration.

## YOLO Model Logic (Suggested Implementation)

This is where the YOLO model runs, actually making the predictions.

- Gather each frame prediction (*Results Object*)
- **Extract the numpy array of yellow values (*ie percent yellow*)**
- **Pass each Box Object (*from results object*) and above numpy array into pipeline**
- **Obtain results, then compare against YOLO results. Could override if different**
- Any other YOLO-related logic

## Pipeline Flow

The pipeline follows these high-level steps:

1. Pulls out the coordinates from the bounding box object
2. Using those coordinates, slices out the subset from the provided full-frame numpy array of yellow values
3. Extracts potential features using that numpy array (ex, mean, std, etc). These features can be anything, and will need to optimized. See *get_bounding_box_features.py* for current extraction ideas.
4. Passes these arguments into a helper model, which generates a prediction of chicks within a given bounding box
5. Returns the bounding box
