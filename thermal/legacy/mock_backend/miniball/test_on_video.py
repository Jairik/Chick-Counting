import cv2
from ultralytics import solutions

video_path = 'C:/Users/anye forti/Desktop/2025 SPRING/425 COSC/YOLO_TESTING/miniball_2tennis.MOV'
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), f"Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT,
    cv2.CAP_PROP_FPS
))

line_points = [(100, 750), (1600, 750)]

video_writer = cv2.VideoWriter(
    "output_with_miniballs.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)
assert video_writer.isOpened(), "Error: video writer failed to open!"

counter = solutions.ObjectCounter(
    show=True,
    region=line_points,
    model="C:/Users/anye forti/Desktop/2025 SPRING/425 COSC/YOLO_TESTING/Chick-Counting/backend/mock_backend/miniball/runs/detect/train7/weights/best.pt",
    line_width=2,
)

try:
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Finished processing the video.")
            break

        im0 = counter.count(im0)
        video_writer.write(im0)

except KeyboardInterrupt:
    print("stopped by user")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
