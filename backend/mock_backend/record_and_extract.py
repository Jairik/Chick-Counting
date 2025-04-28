import cv2
import time
import os

# camera presets
CAM_INDEX = 1
camera_modes = {
    "ultra-fast": {"w": 640, "h": 360, "fps": 260},
    "standard": {"w": 1280, "h": 720, "fps": 120},
    "high-res": {"w": 1920, "h": 1080, "fps": 60}
}

mode = "ultra-fast"
s = camera_modes[mode]

video_file = "samplevid_1.mp4" # where video will be saved
output_folder = "C:/Users/anye forti/Desktop/2025 SPRING/425 COSC/Demo_Extracting" # where frames will be extracted to
frame_interval = 20 # save every 20th frame

# ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
assert cap.isOpened(), "Error reading online camera"

cap.set(cv2.CAP_PROP_FRAME_WIDTH, s["w"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s["h"])
cap.set(cv2.CAP_PROP_FPS, s["fps"])

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
assert video_writer.isOpened(), "Error: video writer failed to open!"

# compute how long to wait in between frames
frame_interval_seconds = 1.0 / fps

frame_count = 0
saved_count = 0

try:
    while cap.isOpened():
        start = time.time()
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        video_writer.write(im0)

        # save frame every N frames
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count}.jpg")
            cv2.imwrite(filename, im0)
            saved_count += 1

        frame_count += 1

        # throttle to target desired fps
        elapsed = time.time() - start
        wait = frame_interval_seconds - elapsed
        if wait > 0:
            time.sleep(wait)

except KeyboardInterrupt:
    print("Ctrl-C detected. Exiting while loop and saving video...")

finally:
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Program terminated successfully. Video saved as '{video_file}', frames saved to '{output_folder}'. Total frames saved: {saved_count}")
