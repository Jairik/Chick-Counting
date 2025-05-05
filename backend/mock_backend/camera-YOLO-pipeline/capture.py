# capture.py

import cv2
import os
import time

def capture_to_disk(frame_dir, s, stop_event, frame_counter):
    os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(s["index"], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, s["w"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s["h"])
    cap.set(cv2.CAP_PROP_FPS, s["fps"])

    print(f"[CAPTURE] Started at {s['w']}x{s['h']} @ {s['fps']} FPS")

    frame_count = 0
    start_time = time.time()
    try:
        while not stop_event.is_set():
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                filename = os.path.join(frame_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(filename, frame)
                frame_count += 1
            except KeyboardInterrupt:
                print("[CAPTURE] Ctrl+C detected during capture. Stopping gracefully...")
                break
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        cap.release()
        frame_counter.value = frame_count
        print(f"[CAPTURE] Done. {frame_count} frames saved.")
        print(f"[CAPTURE] Recording time: {elapsed:.2f} seconds.")
