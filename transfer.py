import torch

yolov5_weights_path = r'C:\Users\fahee\Ball\yolov5\runs\train\yolov5s-custom5\weights\best.pt'

# Load YOLOv5 weights
try:
    yolov5_state_dict = torch.load(yolov5_weights_path, map_location='cpu')
    print("YOLOv5 weights keys:", yolov5_state_dict.keys())
except Exception as e:
    print(f"Error loading YOLOv5 weights: {e}")
