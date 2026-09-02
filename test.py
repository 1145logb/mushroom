
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO("yolov8n.pt")  # 下载并加载 YOLOv8 nano 模型

# 打印模型信息
print(model)
