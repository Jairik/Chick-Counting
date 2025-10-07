#### Readme for backend of chick-counting project.  --  Created 2/12/25 12:53pm by Logan Kelsch

# This Backend will contain the following:

## Backend Phases (3):
-  1) The first phase will consist of implementation of all desired functions, will full functionality and usability within a set of .ipynb files. Model design, training, and experimentation will conclude this phase.
-  2) The second phase will consist of implementaiton of end to end functionality, where raw data to trained model can be done in one step/click, as well as raw data to model evaluation can be done in one step/click.
-  3) The third phase will consist of seamless continuity of the backend, frontend, and hardware to allow for live-stream-data to model output/evaluation in one step/click.

## Backend Files:
-  1) Feature Usage file, for designing and applying transformations to provided data. (Ex: custom time-series formation, LSTM formation, frame2frame clustering, grouping, compressing, rotating, etc.)
-  2) Data Processing file, for data loading, restructing (w/ use of file no.1), and saving with large collection of parameters to use.
-  3) Model Creation/Training file, for designing a model and experimenting with different methods of training.
-  4) Model Evaluation file, for ease of displaying/visualizing performances of a given model on given data.
-  5) Model-Class python file, for mounting of a model allowing for ease of saving, loading, training, and predicting.
-  6) Model distillation file, for implementation of a distillation process or gradient tape, assuming the later phases will demand a tighter window of space-matter and speed-matter.
-  7) Utilities file, for general functionality and misc functionality usage across all files.
-  8) Frontend and Hardware continuity file, for connection of all backend application to all frontend or hardware.
-  9) End-to-end file. This file may not be necessary, but will contain functionality compiling different sections of functionality for user continuity and file 8 usage.

## Considerations of 3/5/2025 week.
- Identify fastest background subtraction methods.
- Consider this processing pipeline:
-   -   Redundant area subtraction
-   -   Redundant color subtraction
-   -   Color contrasting, SWAPPABLE WITH NEXT ITEM
-   -   Color simplification
- .
- To minimize computation or minimize cluster detecting,
- Consider lightweight tracking methods as detection substitution after 'easy-area' detection such as:
-   -   optical flow
-   -   simple centroid tracking
- .
- Discovering low computation detection methods:
-   -   K-means - assuming we can bring in last frame detections as predicted count for next frame
-   -   DBSCAN - pairwise distance calculation must be optimized if used
-   -   complete and/or single linkage - must be heavily optimized if used, since time complexity is O(N^2)
-   -   connected component labeling - O(N) complexity, look into this for sure
-   -   contour detection - O(N) complexity, look into this for sure


## for backend section of presentation of 3/5 week.
-   show client's desired direction for the model and directions to take
-   introduce YOLO
-   show yolo + cole's attempts on chick videos
-   show yolo attempt on fish video
-   make illustration for how image detection + counter model works with code snippet
-   introduce difficulties and the attempt to prepare a wide base for various approaches 
-   -   counter mount, allowing capacity for shown examples of other model types
-   go onto showing yolo training code and parameters if time excess