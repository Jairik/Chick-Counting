import cv2
import os
import time
from ultralytics import solutions
from config import FRAME_DIR, OUTPUT_VIDEO

def process_frame(frame_dir, counter, video_writer):
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith('.jpg'))
    if not frames:
        return False
    frame_path = os.path.join(frame_dir, frames[0])
    frame = cv2.imread(frame_path)
    if frame is None:
        os.remove(frame_path)
        return True
    result = counter.count(frame)
    video_writer.write(result)
    os.remove(frame_path)
    return True

def process_from_disk(frame_dir, w, h, fps, stop_event, frame_counter):
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    counter = solutions.ObjectCounter(
        show=True,
        region=[(100, 200), (100, 600)],
        model="yolo11n.pt",
        line_width=2,
    )

    processed = 0
    print(f"[YOLO] Waiting for frames in {FRAME_DIR}...")

    try:
        while True:
            if not process_frame(FRAME_DIR, counter, video_writer):
                if stop_event.is_set():
                    print("[YOLO] Queue is empty and stop_event is set. Exiting.")
                    break
                time.sleep(0.01)
            else:
                processed += 1
    except KeyboardInterrupt:
        print("[YOLO] Ctrl+C detected — draining remaining frames...")
        while process_frame(FRAME_DIR, counter, video_writer):
            processed += 1
    finally:
        video_writer.release()
        print(f"[YOLO] Done. {processed} frames processed.")
        print(f"[CAPTURE] {frame_counter.value} frames recorded.")
