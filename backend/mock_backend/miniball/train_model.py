from ultralytics import YOLO

# loads yolo11 model
model = YOLO('yolo11n.pt')

# trains model with miniball dataset
model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640
)
