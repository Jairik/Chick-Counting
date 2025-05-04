import cv2
ew, eh, external_fps = 1920, 1080, 60  # External RGB camera dimensions (known)

external_rgb_filename = "external-rgb-test" + ".m264"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec

def external_capture_rgb():
    cap = cv2.VideoCapture(0)  # Open the default webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ew)  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, eh)  # Set height
    cap.set(cv2.CAP_PROP_FPS, external_fps)  # Set FPS

    if not cap.isOpened():
        print("External RGB camera could not be opened")
        return

    out = cv2.VideoWriter(external_rgb_filename, fourcc, external_fps, (ew, eh))

    try:
        while True:  # TESTING
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from external RGB camera")
                break

            out.write(frame)  # Write the frame to the video file

            if True:  # TESTING 
                cv2.imshow("External RGB Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("External RGB video capture interrupted")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("External RGB Camera resources released")
        
external_capture_rgb()