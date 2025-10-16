# TODO - Awaiting access to Brennen's YOLO model, refine once access is granted

from utils.get_attributes import get_bounding_boxes
from utils.path_conversions import get_source
from ultralytics import YOLO

''' Return the local video source for testing the thermal model 
NOTE: This is under the assumption that you are running under WSL on Windows 10/11. If not, replace with the explicit path
'''
def get_local_source():
    src: str = ""
    try:
        src = get_source()
    except FileNotFoundError as e:
        print(e)
        raise e
    print(f"Using video source: {src}")
    return src
    
''' Load the pretrained YOLO model and generate results '''
model = YOLO("../../yolo11n.pt")
results = model(source=get_local_source(), stream=True, visualize=True, 
                show=True, show_boxes=True, conf=0.25, save_txt=True, save=True)

# DEBUGGING
print(results)

# Print the boxes detected in the video
boxes = get_bounding_boxes(results)
print("Printing bounding boxes:\n")
print(boxes)

