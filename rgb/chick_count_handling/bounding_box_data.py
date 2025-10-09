"""
This script processes a video with a YOLO model to store boxes that cross the counter line. It outputs:

1. YOLO output video.
2. A CSV file logging each new crossing event:
   - Frame number, object ID, bounding box coordinates, area, and a clickable
     link to a saved snapshot.
3. A folder of PNG snapshots corresponding to each detected crossing.
"""

import os
import cv2
import csv
from ultralytics import solutions

# ——— PATH CONFIG —————————————————————————————————————————————————
VIDEO_PATH        = ""
OUTPUT_VIDEO_PATH = ""
CROSS_CSV_PATH    = ""
MODEL_PATH        = ""

# ——— PREP OUTPUT FOLDERS & CSV ————————————————————————————————————
os.makedirs(os.path.dirname(CROSS_CSV_PATH), exist_ok=True)
snapshot_dir = os.path.join(os.path.dirname(CROSS_CSV_PATH), "snapshots")
os.makedirs(snapshot_dir, exist_ok=True)

cross_file   = open(CROSS_CSV_PATH, "w", newline="")
cross_writer = csv.writer(cross_file)
cross_writer.writerow([
    "frame", "main_id",
    "x1", "y1", "x2", "y2", "area",
    "snapshot_link"
])

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
    show=True,
    region=[(320, 560), (1820, 540)],
    model=MODEL_PATH,
    line_width=2,
    device="cuda:0" #COMMENT OUT IF NOT USING GPU
)

def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

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

        res       = counter(frame)
        annotated = res.plot_im
        writer_vid.write(annotated)

        new_ids   = set(counter.counted_ids) - prev_ids
        in_events = counter.in_count - prev_in
        rec       = 0

        # capture boxes and IDs
        boxes = [tuple(map(int, b)) for b in counter.boxes]
        tids  = counter.track_ids

        # log new crossings
        for tid in new_ids:
            idx       = tids.index(tid)
            main_box  = boxes[idx]
            main_area = bbox_area(main_box)

            # save snapshot and link
            img_name    = f"frame{frame_idx:05d}_id{tid}.png"
            img_path    = os.path.join(snapshot_dir, img_name)
            cv2.imwrite(img_path, annotated)
            link_formula = f'=HYPERLINK("file:///{img_path}", "{img_name}")'

            # write main crossing
            cross_writer.writerow([
                frame_idx, tid,
                *main_box, main_area,
                link_formula
            ])

            rec += 1
            if rec >= in_events:
                break

except KeyboardInterrupt:
    print("Interrupted—saving outputs so far.")

finally:
    cap.release()
    writer_vid.release()
    cv2.destroyAllWindows()

    cross_file.close()

    print(f"Done up to frame {frame_idx}.")
    print("Annotated video:", OUTPUT_VIDEO_PATH)
    print("Crossings CSV:", CROSS_CSV_PATH)
    print("Snapshots saved in:", snapshot_dir)
