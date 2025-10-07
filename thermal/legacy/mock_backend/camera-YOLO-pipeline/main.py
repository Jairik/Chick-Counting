# main.py

import os
import time
import multiprocessing as mp
from capture import capture_to_disk
from processing import process_from_disk
from config import camera_modes, ACTIVE_MODE, CAM_INDEX, FRAME_DIR

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    s = camera_modes[ACTIVE_MODE]
    s["index"] = CAM_INDEX
    s["fps"] = 60

    stop_event = mp.Event()
    frame_counter = mp.Value('i', 0)

    print(f"[MAIN] Selected mode '{ACTIVE_MODE}' -> {s['w']}x{s['h']} @ ~{s['fps']} FPS")
    print("[MAIN] Launching capture and YOLO processes...")

    cap_proc = mp.Process(target=capture_to_disk, args=(FRAME_DIR, s, stop_event, frame_counter))
    yolo_proc = mp.Process(target=process_from_disk, args=(FRAME_DIR, s["w"], s["h"], s["fps"], stop_event, frame_counter))

    cap_proc.start()
    yolo_proc.start()

    try:
        while cap_proc.is_alive():
            if not os.path.exists(FRAME_DIR):
                time.sleep(0.5)
                continue
            remaining = len([f for f in os.listdir(FRAME_DIR) if f.endswith('.jpg')])
            print(f"[MAIN] Queue size (on disk): {remaining} frames")
            time.sleep(2)
    except KeyboardInterrupt:
        print("[MAIN] Ctrl+C detected. Stopping camera only. YOLO will continue.")
        stop_event.set()

    cap_proc.join()
    print("[MAIN] Camera process finished. Waiting for YOLO to drain queue...")

    yolo_proc.join()

    try:
        if os.path.exists(FRAME_DIR) and not os.listdir(FRAME_DIR):
            os.rmdir(FRAME_DIR)
            print(f"[MAIN] Cleaned up temporary folder: {FRAME_DIR}")
        else:
            print(f"[MAIN] Folder '{FRAME_DIR}' not empty, skipped deletion.")
    except Exception as e:
        print(f"[MAIN] Failed to delete temp folder: {e}")

    print(f"[MAIN] All processes complete. Final video saved to: {camera_modes[ACTIVE_MODE]['w']}x{camera_modes[ACTIVE_MODE]['h']}")
