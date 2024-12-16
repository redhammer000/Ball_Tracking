from ultralytics import YOLO
import torch


yolov8_weights_path = 'yolov8n.pt'

# Initialize YOLOv8 model
yolov8_model = YOLO(yolov8_weights_path)



config = {
    'data': 'data.yaml',
    'epochs': 50,
    'imgsz': 640,
    'batch': 8,
    'device': 'cpu',
    'workers': 4,
    'lr0': 0.01,
    'lrf': 0.2,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'optimizer': 'SGD',
}

# Fine-tune YOLOv8 model
yolov8_model.train(
    data=config['data'],
    epochs=config['epochs'],
    imgsz=config['imgsz'],
    batch=config['batch'],
    device=config['device'],
    workers=0
)


