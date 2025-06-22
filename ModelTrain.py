import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="C:/Users/User/Desktop/deeplearning/model/data.yaml", epochs=50, imgsz=640, workers=0)
