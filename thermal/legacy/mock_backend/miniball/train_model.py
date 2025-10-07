from ultralytics import YOLO

# loads yolo11 model
model = YOLO('C:/Users/anye forti/Desktop/2025 SPRING/425 COSC/YOLO_TESTING/Chick-Counting/backend/mock_backend/miniball/yolo11n.pt')

# trains model with miniball dataset
model.train(
    data='C:/Users/anye forti/Desktop/2025 SPRING/425 COSC/YOLO_TESTING/Chick-Counting/backend/mock_backend/miniball/data.yaml',
    epochs=100,
    imgsz=640
)