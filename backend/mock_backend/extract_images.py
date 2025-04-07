# this program uses OpenCV to extract frames from a video.
# also uses os module to handle where frames get saved
# currently being used for labeling objects with Roboflow
import cv2
import os


video_file = "C:/Users/anye forti/Desktop/2025 SPRING/425 COSC/IMG_3222.MOV" # may be scaled up to use live video footage from raspberry pi
output_folder = "C:/Users/anye forti/Desktop/2025 SPRING/425 COSC/Demo_Extracting" # directory for images to be saved to
frame_interval = 20 # so were not saving every single frame
assert os.path.exists(video_file), "error: video file not found"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_file)
frame_count = 0
saved_count = 0
assert cap.isOpened(), "error opening video file"

while cap.isOpened():
	success, frame = cap.read()
	if not success:
		print("video finished")
		print("number of saved frames:", saved_count)
		break

	# save frame as image in chosen directory
	if frame_count % frame_interval == 0:
		filename = os.path.join(output_folder, f"frame_{saved_count}.jpg")
		cv2.imwrite(filename, frame)

		saved_count += 1

	frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("program terminated successfully")