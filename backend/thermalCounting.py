import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO

def get_line_from_video_frame(frame):
    frame_height, frame_width = frame.shape[:2]

    # Draw a horizontal line across the middle of the frame
    line_start = (frame_width, frame_height // 2)
    line_end = (0, frame_height // 2)
    return [line_start, line_end]

def chick_counting(video_path, output_path, line_points):

    # Grab a sample frame so we know video size
    generator = sv.get_video_frames_generator(video_path)
    frame = next(generator)

    # Set up video writer with same FPS/size as input
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
    if not out.isOpened():
        print("Error: Could not open video writer")
        return

    # Init tracker and helpers
    byte_tracker = sv.ByteTrack()
    trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

    # Create the counting line
    line_zone = sv.LineZone(start=sv.Point(*line_points[0]), end=sv.Point(*line_points[1]))

    # Load custom YOLO model (trained on chicks only)
    model = YOLO("E:\\25 Summer Research\\new_iron.pt")

    # Annotators for boxes + labels
    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
    LABEL_ANNOTATOR = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=1,
        text_color=sv.Color.BLACK
    )

    frame_count = 0
    total_count = 0
    all_counted_ids = set()  # keep track of already-counted trackers

    try:
        generator = sv.get_video_frames_generator(video_path)

        for frame in generator:
            frame_count += 1
            print(f"Processing frame {frame_count}")

            # Run YOLO on frame
            results = model(frame)[0]

            # Convert results to supervision Detections
            detections = sv.Detections.from_ultralytics(results)

            # Update tracker with detections
            detections = byte_tracker.update_with_detections(detections)
            print("Tracker IDs this frame:", detections.tracker_id)

            # See if any trackers crossed the line
            crossed_in_flags, crossed_out_flags = line_zone.trigger(detections)

            # Only count new IDs that cross "in"
            for i, crossed in enumerate(crossed_in_flags):
                if crossed:
                    tracker_id = detections.tracker_id[i]
                    if tracker_id is not None and tracker_id not in all_counted_ids:
                        total_count += 1
                        all_counted_ids.add(tracker_id)
                        print(f"New Chick crossed the line! ID {tracker_id}, Total count: {total_count}")

            # Sensitivity for declaring a box as "nested"
            # e.g. 0.9 means inner must have at least 90% of its area inside outer
            NESTED_THRESHOLD = 0.9  

            contained_indices = set()
            boxes = detections.xyxy

            for i, outer in enumerate(boxes):
                x1o, y1o, x2o, y2o = outer
                outer_area = max(0, (x2o - x1o)) * max(0, (y2o - y1o))

                for j, inner in enumerate(boxes):
                    if i == j:
                        continue
                    x1i, y1i, x2i, y2i = inner
                    inner_area = max(0, (x2i - x1i)) * max(0, (y2i - y1i))

                    # Intersection box
                    inter_x1 = max(x1o, x1i)
                    inter_y1 = max(y1o, y1i)
                    inter_x2 = min(x2o, x2i)
                    inter_y2 = min(y2o, y2i)

                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

                    # Ratio of inner covered by outer
                    if inner_area > 0 and (inter_area / inner_area) >= NESTED_THRESHOLD:
                        contained_indices.add(j)


            # Assign labels + colors depending on nesting
            labels = []
            colors = []
            for i, tracker_id in enumerate(detections.tracker_id):
                if i in contained_indices:
                    labels.append(f"#{tracker_id} nested")
                    colors.append(sv.Color.RED)
                else:
                    labels.append(f"#{tracker_id} chick")
                    colors.append(sv.Color.GREEN)

            # Draw tracker trails
            annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)

            # Draw bounding boxes manually with chosen colors
            for i, box in enumerate(detections.xyxy):
                color = colors[i] if i < len(colors) else sv.Color.GREEN
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color.as_bgr(), 2)

            # Draw labels
            annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)

            # Draw the counting line
            cv2.line(annotated_frame, line_points[0], line_points[1], (0, 0, 255), 2)

            # Overlay total count
            cv2.putText(
                annotated_frame,
                f'Total Count: {total_count}',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Write out annotated frame
            out.write(annotated_frame)

    except Exception as e:
        print(f"Error during processing: {e}")

    finally:
        # Clean up writer and windows
        out.release()
        cv2.destroyAllWindows()
        print(f"Processing complete. Processed {frame_count} frames.")
        print(f"Final total count: {total_count}")
        print(f"LineZone internal count (for reference): in={line_zone.in_count}, out={line_zone.out_count}")


if __name__ == "__main__":
    import tkinter as tk
    from tkinter.filedialog import askopenfilename, askdirectory
    tk.Tk().withdraw()

    # Pick input video + output folder with file dialogs
    SOURCE_VIDEO_PATH = askopenfilename()
    print("User chose:", SOURCE_VIDEO_PATH)

    folder_path = askdirectory()
    print("Output folder:", folder_path)

    # Build output filename
    filename_no_ext = SOURCE_VIDEO_PATH.split('/')[-1].rsplit('.', 1)[0]
    OUTPUT_PATH = f"{folder_path}/{filename_no_ext}-outputfile(colored).mp4"
    print("Output path:", OUTPUT_PATH)

    # Grab a frame to define the line
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video")
        exit()
    cap.release()

    line_points = get_line_from_video_frame(frame)
    print("Line points:", line_points)

    # Only run if line points are valid
    if len(line_points) == 2:
        chick_counting(SOURCE_VIDEO_PATH, OUTPUT_PATH, line_points)
    else:
        print("Error: Not enough points to define the counting line.")