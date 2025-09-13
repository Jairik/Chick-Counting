# TODO - Awaiting access to Brennen's YOLO model, refine once access is granted

from pathlib import Path
from utils.get_attributes import get_bounding_boxes
from ultralytics import YOLO

''' Return the local video source for testing the thermal model '''
def get_local_source():
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parents[2]  # Chick-Counting/
    return str(ROOT / "data" / "Brennen-Thermal-Video" / "Top_Belt(Iron)_01.mp4")  # Return the source
    
''' Load the pretrained YOLO model and generate results '''
model = YOLO("../../yolo11n.pt")
results = model(source=get_local_source(), stream=True, visualize=True, 
                show=True, show_boxes=True, conf=0.25, save_txt=True, save=True)

# Print the boxes detected in the video
boxes = get_bounding_boxes(results)
print("Printing bounding boxes:\n")
print(boxes)

