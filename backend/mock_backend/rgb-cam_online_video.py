#Updated USB RGB camera code to support desired fps in recording

import cv2
import time

# camera presets
CAM_INDEX = 1
camera_modes = {
	"ultra-fast" : {"w": 640, "h": 360, "fps": 260},
	"standard" : {"w": 1280, "h": 720, "fps": 120},
	"high-res" : {"w": 1920, "h": 1080, "fps": 60}
}

mode = "ultra-fast"
s = camera_modes[mode]

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
assert cap.isOpened(), "Error reading online camera"

cap.set(cv2.CAP_PROP_FRAME_WIDTH, s["w"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s["h"])
cap.set(cv2.CAP_PROP_FPS, s["fps"])

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("samplevid_1.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
assert video_writer.isOpened(), "Error: video writer failed to open!"

# compute how long to wait in between frames
frame_interval = 1.0 / fps

try:
	while cap.isOpened():
		start = time.time()
		success, im0 = cap.read()
		if not success:
			print("Video frame is empty or video processing has been successfully completed.")
			break

		video_writer.write(im0)

		# throttle to target desired fps
		elapsed = time.time() - start
		wait = frame_interval - elapsed
		if wait > 0:
			time.sleep(wait)

except KeyboardInterrupt:
	print("Ctrl-C detected. Exiting while loop and saving video...")

cap.release()
video_writer.release()
cv2.destroyAllWindows()