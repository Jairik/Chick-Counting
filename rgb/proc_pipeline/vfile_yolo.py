"""
This script processes video with YOLO object detection and applies weighted counting based on bounding box size checks.

Input:
- chick video to be tested
- custom chick-trained model
Output:
- annotated chick video with both YOLO and weighted counts
"""

import cv2
from ultralytics import solutions
from bbox_utils import BoundingBox, check_and_count

# ——— PATH CONFIG —————————————————————————————————————————————————
VIDEO_PATH        = "C:/Users/anye forti/Desktop/PERDUE FARMS/perdue_rgb_video2_061725.mp4"
OUTPUT_VIDEO_PATH = "C:/Users/anye forti/Desktop/2025 SPRING/425 COSC/YOLO_TESTING/YOLO Videos/pipeline_test.mp4"
MODEL_PATH        = "C:/Users/anye forti/Desktop/PERDUE FARMS/chick-test-1/fold_2/runs/detect/train/weights/best.pt"
# —————————————————————————————————————————————————————————————————

# draws counter overlay on the frame showing both YOLO and weighted counts.
def draw_counter_overlay(frame, yolo_count, weighted_count):
    h, w = frame.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_color = (0, 255, 0)
    bg_color = (0, 0, 0)
    padding = 10
    line_spacing = 40
    
    x = w - 20
    y = 20
    
    text_lines = [
        f"YOLO Count: {yolo_count}",
        f"Weighted Count: {weighted_count}"
    ]
    
    max_width = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in text_lines])
    bg_height = len(text_lines) * line_spacing + padding * 2

    x_left = x - max_width - padding * 2

    cv2.rectangle(frame, 
                (x_left, y - padding), 
                (x, y + bg_height - padding), 
                bg_color, -1)

    (_, text_height), _ = cv2.getTextSize(text_lines[0], font, font_scale, thickness)
    first_line_y = y + padding + text_height

    for i, line in enumerate(text_lines):
        y_pos = first_line_y + (i * line_spacing)
        cv2.putText(frame, line, (x_left + padding, y_pos), font, font_scale, text_color, thickness)



# ——— SETUP VIDEO + COUNTER ————————————————————————————————————————
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Could not open input video"

w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
writer_vid = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))
assert writer_vid.isOpened(), "Could not open output video"

counter = solutions.ObjectCounter(
    show=False,
    region=[(320, 560), (1820, 540)],
    model=MODEL_PATH,
    line_width=2,
    device="cuda:0",
)

# ——— TRACKING VARIABLES ———————————————————————————————————————————
total_weighted_count = 0

# ——— PROCESS VIDEO STREAM ————————————————————————————————————————
frame_idx = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        prev_ids = set(counter.counted_ids)
        prev_in  = counter.in_count

        res       = counter.count(frame)
        annotated = res
        
        draw_counter_overlay(annotated, counter.in_count, total_weighted_count)
        
        cv2.imshow("YOLO Object Counter", annotated)
        cv2.waitKey(1)
        
        writer_vid.write(annotated)

        new_ids   = set(counter.counted_ids) - prev_ids
        in_events = counter.in_count - prev_in
        rec       = 0

        boxes = [tuple(map(int, b)) for b in counter.boxes]
        tids  = counter.track_ids

        for tid in new_ids:
            idx = tids.index(tid)
            box_coords = boxes[idx]
            
            bbox = BoundingBox(*box_coords)
            
            count_value = check_and_count(bbox)
            total_weighted_count += count_value
            
            # error checking
            area = bbox.calculate_area()
            print(f"Frame {frame_idx}: ID {tid} crossed (count: {count_value}, area: {area})")

            rec += 1
            if rec >= in_events:
                break

except KeyboardInterrupt:
    print("Interrupted—saving outputs so far.")

finally:
    cap.release()
    writer_vid.release()
    cv2.destroyAllWindows()

    print(f"\nDone processing {frame_idx} frames.")
    print(f"Total YOLO count: {counter.in_count}")
    print(f"Total weighted count: {total_weighted_count}")
    print("Annotated video:", OUTPUT_VIDEO_PATH)