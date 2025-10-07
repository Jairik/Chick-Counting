"""
This script provides a manual video review tool for verifying and counting 
crossings frame-by-frame. It lets the user step through a video interactively 
and record counts to a CSV file.

Features:
1. Displays the current frame with overlayed frame index and running count.
2. Keyboard controls:
   - N: move forward by STEP_SIZE frames
   - P: move backward by STEP_SIZE frames
   - C: record a new crossing count at current frame
   - U: undo the last count
   - Q: quit the program
3. Saves results to a CSV file with columns: frame_index, cross_count.
"""

import cv2
import csv
import os

# ─── CONFIG ────────────────────────────────────────────────────────────────────
VIDEO_PATH  = ""
OUTPUT_CSV  = ""
STEP_SIZE   = 2
# ───────────────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video:", VIDEO_PATH)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx    = 0
    cross_count  = 0
    history      = []  # for undo

    # prepare CSV
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
    csv_file = open(OUTPUT_CSV, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(["frame_index", "cross_count"])

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print("Reached end or cannot read frame.")
            break

        # overlay frame index & count
        cv2.putText(frame, f"Frame: {frame_idx+1}/{total_frames}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Count: {cross_count}", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Verify Crossings", frame)
        key = cv2.waitKey(0) & 0xFF   # wait for a key

        if key in (ord('n'), ord('N')):           # next STEP_SIZE frames
            frame_idx = min(frame_idx + STEP_SIZE, total_frames - 1)
        elif key in (ord('p'), ord('P')):         # previous STEP_SIZE frames
            frame_idx = max(frame_idx - STEP_SIZE, 0)
        elif key in (ord('c'), ord('C')):         # count a crossing
            cross_count += 1
            history.append(cross_count)
            writer.writerow([frame_idx, cross_count])
            print(f"Counted at frame {frame_idx}: total={cross_count}")
        elif key in (ord('u'), ord('U')):         # undo last count
            if history:
                history.pop()
                cross_count = history[-1] if history else 0
                print(f"Undo: back to {cross_count}")
            else:
                print("Nothing to undo.")
        elif key in (ord('q'), ord('Q')):         # quit
            break
        else:
            print(f"Keys: n=+{STEP_SIZE}, p=-{STEP_SIZE}, c=count, u=undo, q=quit")

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("Finished. Counts saved to", OUTPUT_CSV)

if __name__ == "__main__":
    main()
