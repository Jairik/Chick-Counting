import cv2

from ultralytics import solutions

# Open the webcam
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading online camera"

# Get video properties: width, height, and frames per second (fps)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define points for a line or region of interest in the video frame
line_points = [(100, 200), (100, 600)]  # Line coordinates

# Initialize the video writer to save the output video
video_writer = cv2.VideoWriter("samplevid0.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
assert video_writer.isOpened(), "Error: video writer failed to open!"

# Initialize the Object Counter with visualization options and other parameters
counter = solutions.ObjectCounter(
    show=True,  # Display the image during processing
    region=line_points,  # Region of interest points
    model="yolo11n.pt",  # Ultralytics YOLO11 model file
    line_width=2,  # Thickness of the lines and bounding boxes
)

# added exception to catch keyboard interrupt to end stream
try:
	while cap.isOpened():
		success, im0 = cap.read()
		if not success:
			print("Video frame is empty or video processing has been successfully completed.")
			break

		# Use the Object Counter to count objects in the frame and get the annotated image
		im0 = counter.count(im0)

		# Write the annotated frame to the output video
		video_writer.write(im0)
except KeyboardInterrupt:
	print("Ctrl-C detected. Exiting while loop and saving video...")

# Release the video capture and writer objects
cap.release()
video_writer.release()
#print("width:", w)
#print("height:", h)
#print("frames per sec:", fps)

# Close all OpenCV windows
cv2.destroyAllWindows()