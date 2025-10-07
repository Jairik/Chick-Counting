# config.py

CAM_INDEX = 1

FRAME_DIR = "C:/Users/anye forti/Desktop/2025 SPRING/425 COSC/YOLO_TESTING/frames"
OUTPUT_VIDEO = "C:/Users/anye forti/Desktop/2025 SPRING/425 COSC/YOLO_TESTING/rgb_yolo_pipeline_output.mp4"

camera_modes = {
    "ultra-fast": {"w": 640, "h": 360},
    "standard": {"w": 1280, "h": 720},
    "high-res": {"w": 1920, "h": 1080},
}

ACTIVE_MODE = "standard"