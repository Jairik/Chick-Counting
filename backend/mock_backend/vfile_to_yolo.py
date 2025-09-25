import cv2
from ultralytics import solutions

# Video file path (replace with your actual path)
VIDEO_PATH = ""

# Open the video file instead of the webcam
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Error: could not open video file"

# Get video properties: width, height, and frames per second (fps)
w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS
))

# Define points for a line or region of interest in the video frame
line_points = [(100, 200), (100, 600)]  # Line coordinates

# Initialize the video writer to save the output video
video_writer = cv2.VideoWriter("crazy_test.mp4",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (w, h))
assert video_writer.isOpened(), "Error: video writer failed to open!"

# Initialize the Object Counter with visualization options and other parameters
counter = solutions.ObjectCounter(
    show=True,                   # Display the image during processing
    region=line_points,         # Region of interest points
    model="",         # Ultralytics YOLOv11 model file
    line_width=2,               # Thickness of the lines and bounding boxes
)

# Process the video file frame by frame
try:
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("End of video or error reading frame.")
            break

        # Use the Object Counter to count objects and get the annotated frame
        im0 = counter.count(im0)

        # Write the annotated frame to the output video
        video_writer.write(im0)

except KeyboardInterrupt:
    print("Ctrl-C detected. Exiting and saving video...")

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()
