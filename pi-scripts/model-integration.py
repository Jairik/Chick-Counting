''' Re-implementation of code found in the backend folder, using libcamera for live video integration '''

# Model Imports
import ultralytics
ultralytics.checks()

# Testing libcamera dependency
import libcamera
# help(libcamera)  # Display message if installed properly

# Input Capture Methods Input
from libcamera import controls
from picamera2 import Picamera2
from ultralytics import solutions
from ultralytics.solutions import ObjectCounter
import cv2

# Initialize & start libcamera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1920, 1080), "format":"RGB888"})
picam2.configure(config)
picam2.start()

print("Camera Initialized")

# Defining video output file name
OUTPUT_FILE_NAME = "output-test.avi"

# Defining video output properties (should match input)
w, h, fps = 1920, 1080, 30

# Define points for a line or region of interest in the video frame
line_points = [(400, 100), (400, 620)]  # Line coordinates

# Initialize the video writer to save the output video
video_writer = cv2.VideoWriter(OUTPUT_FILE_NAME, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

model = "models/yolo11n.pt"

print("Video Writer Initialed\nTo exit out of loop, press q")

# Initialize the Object Counter with visualization options and other parameters
counter = solutions.ObjectCounter(
    show=True,  							# Display the image during processing
    region=line_points,  					# Region of interest points
    model="models/yolo11n.pt",  			# Ultralytics YOLO11 model file
    line_width=2,  							# Thickness of the lines and bounding boxes
)

# Process live video frames in a loop
while True:  # Conditional can be modified once more capturing details are known
    
    # Capture current frame from libcamera
    frame = picam2.capture_array()
    
    # Convert from RGB to BRG (opencv format)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Use the Object Counter to count objects in the frame and get the annotated image
    processed_frame = counter.count(frame_bgr)

    # Write the annotated frame to the output video
    video_writer.write(processed_frame)
    
    # Display the frame
    cv2.imshow("Object Detection", processed_frame)
    
    # Exit on the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
picam2.stop()
video_writer.release()
cv2.destroyAllWindows()
