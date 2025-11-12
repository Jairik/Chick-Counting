"""
This script processes video with YOLO object detection and applies weighted counting based on bounding box size checks. This weighted counting is stored in separate spreadsheet file for additional analysis.

Input:
- chick video to be tested
- custom chick-trained model
- file paths for count snapshots and xlsx data file
Output:
- annotated chick video with both YOLO and weighted counts
- xlsx data file for weighted counts
"""

import cv2
import os
import pandas as pd
from ultralytics import solutions
from bbox_utils import BoundingBox, check_group, reset_conjoined_state

# ——— PATH CONFIG —————————————————————————————————————————————————
VIDEO_PATH          = r"C:\Users\anye forti\Desktop\PERDUE FARMS\perdue_rgb_video2_061725.mp4"
OUTPUT_VIDEO_PATH   = "C:/Users/anye forti/Desktop/2025 FALL/426 COSC/YOLO_TESTING/YOLO Videos/yolo_count_proc_v2_pt2.mp4"
MODEL_PATH          = r"c:\Users\anye forti\Desktop\PERDUE FARMS\chick-test-1\fold_2\runs\detect\train\weights\best.pt"

ENABLE_XLSX_EXPORT  = True
XLSX_PATH           = "C:/Users/anye forti/Desktop/2025 SPRING/425 COSC/YOLO_TESTING/Chick-Counting/rgb/data/weighted_count_data_v2_pt2.xlsx"
SNAPSHOT_DIR        = "C:/Users/anye forti/Desktop/2025 FALL/426 COSC/YOLO_TESTING/bbox-snapshots2"
SNAPSHOT_NAME_FMT   = "frame{frame:05d}_id{tid}.png"
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

# ——— HELPER FUNCTIONS —————————————————————————————————————————————
def snapshot_name_format(frame_idx, tid):
    return SNAPSHOT_NAME_FMT.format(frame=frame_idx, tid=tid)

def save_snapshot(annotated_frame, ids_to_highlight, id_to_bbox, frame_idx, tid):
    if not ENABLE_XLSX_EXPORT:
        return
    
    snapshot = annotated_frame.copy()
    
    for id in ids_to_highlight:
        box = id_to_bbox.get(id)
        if box is None:
            continue
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 4)

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    file = snapshot_name_format(frame_idx, tid)
    dir = os.path.join(SNAPSHOT_DIR, file)
    cv2.imwrite(dir, snapshot)

def make_snapshot_hyperlink(frame_idx, tid):
    file = snapshot_name_format(frame_idx, tid)
    base = SNAPSHOT_DIR.replace("\\", "/")
    link = f'file:///{base}/{file}'

    return f'=HYPERLINK("{link}", "{file}")'

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

total_weighted_count = 0
main_rows = []
conjoined_rows = []
CHILD_HISTORY = set()

# ——— PROCESS VIDEO STREAM ————————————————————————————————————————
frame_idx = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        reset_conjoined_state()

        prev_ids = set(counter.counted_ids)
        prev_in  = counter.in_count

        annotated = counter.count(frame)
        
        draw_counter_overlay(annotated, counter.in_count, total_weighted_count)
        
        cv2.imshow("YOLO Object Counter", annotated)
        cv2.waitKey(1)
        
        writer_vid.write(annotated)

        new_ids   = set(counter.counted_ids) - prev_ids

        boxes = [tuple(map(int, b)) for b in counter.boxes]
        tids  = counter.track_ids
        frame_bboxes = [BoundingBox(*coords, obj_id=tids[i]) for i, coords in enumerate(boxes)]
        id_to_bbox = {box.obj_id: box for box in frame_bboxes if box.obj_id is not None}

        ordered_new_ids = [tid for tid in tids if tid in new_ids]
        skip_ids = set()

        for tid in ordered_new_ids:
            if tid in CHILD_HISTORY:
                continue

            if tid in skip_ids:
                continue

            idx = tids.index(tid)
            main_first = [frame_bboxes[idx]] + frame_bboxes[:idx] + frame_bboxes[idx+1:]
            meta = check_group(main_first)
            
            if meta.get('was_conjoined'):
                child_id = meta.get('main_id', tid)
                if child_id is not None:
                    CHILD_HISTORY.add(child_id)
                    skip_ids.add(child_id)
                continue

            parent_id = meta.get('main_id', tid)
            child_ids = meta.get('conjoined_ids', [])

            for c in child_ids:
                CHILD_HISTORY.add(c)

            total_weighted_count += meta.get('weighted_count', 0)

            if ENABLE_XLSX_EXPORT:
                ids_to_highlight = [parent_id] + child_ids
                save_snapshot(annotated, ids_to_highlight, id_to_bbox, frame_idx, tid)

                main_rows.append({
                    'frame': frame_idx,
                    'main_id': meta.get('main_id', tid),
                    'area': meta.get('area', None),
                    'weighted_ct': meta.get('weighted_count', 0),
                    'snapshot_link': make_snapshot_hyperlink(frame_idx, tid)
                })

                for child_id in child_ids:
                    conjoined_rows.append({
                        'frame': frame_idx,
                        'main_id': parent_id,
                        'conjoined_id': child_id
                    })

            skip_ids.update(child_ids)
            skip_ids.add(parent_id)

except KeyboardInterrupt:
    print("Interrupted—saving outputs so far.")

finally:
    cap.release()
    writer_vid.release()
    cv2.destroyAllWindows()

    if ENABLE_XLSX_EXPORT:
        os.makedirs(os.path.dirname(XLSX_PATH), exist_ok=True)

        df_main = pd.DataFrame(main_rows, columns=['frame', 'main_id', 'area', 'weighted_ct', 'snapshot_link'])
        df_conj = pd.DataFrame(conjoined_rows, columns=['frame', 'main_id', 'conjoined_id'])

        with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as writer:
            df_main.to_excel(writer, sheet_name="main", index=False)
            df_conj.to_excel(writer, sheet_name="conjoined", index=False)

        print(f"Spreadsheet written: {XLSX_PATH}")

    print(f"\nDone processing {frame_idx} frames.")
    print(f"Total YOLO count: {counter.in_count}")
    print(f"Total weighted count: {total_weighted_count}")
    print(f"Annotated video: {OUTPUT_VIDEO_PATH}")