from ultralytics import YOLO
import cv2
import os
import pandas as pd
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ================= Kalman Filter =================
class KalmanPoint:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self.initialized = False

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])

        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True

        self.kf.correct(measurement)
        prediction = self.kf.predict()

        return int(prediction[0][0]), int(prediction[1][0])


    def predict_only(self):
        prediction = self.kf.predict()
        return int(prediction[0][0]), int(prediction[1][0])

# ================= 模型 =================
model_keypoint_path = r"C:\Users\sywan\Desktop\code\runs\pose\train7\weights\best.pt"
model_keypoint = YOLO(model_keypoint_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

keypoint_names = ['left', 'right', 'root']
records = []
save_count = 0

# 初始化 Kalman（3個點）
kalman_filters = [KalmanPoint() for _ in range(3)]

while True:
    ret, img = cap.read()
    if not ret:
        break

    h, w = img.shape[:2]

    results = model_keypoint.predict(source=img, conf=0.25, verbose=False)

    if (not hasattr(results[0], "keypoints") or 
        results[0].keypoints is None or 
        len(results[0].keypoints.xy) == 0):

        print("⚠️ 無法偵測")
        cv2.imshow("frame", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    keypoints_xy = results[0].keypoints.xy[0].cpu().numpy()
    keypoints_conf = (
        results[0].keypoints.conf[0].cpu().numpy()
        if results[0].keypoints.conf is not None
        else np.ones(len(keypoints_xy))
    )

    obj_conf = float(results[0].boxes.conf[0]) if results[0].boxes is not None else 0.0

    # ================= Kalman 平滑 =================
    smoothed_points = []

    for idx, ((kx, ky), kv) in enumerate(zip(keypoints_xy, keypoints_conf)):
        
        if kv > 0.5:
            sx, sy = kalman_filters[idx].update(kx, ky)
        else:
            sx, sy = kalman_filters[idx].predict_only()

        smoothed_points.append((sx, sy))

        # 畫平滑後的點
        cv2.circle(img, (sx, sy), 10, (0, 255, 0), -1)
        cv2.putText(img, keypoint_names[idx], (sx+10, sy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # ================= 距離（用平滑後） =================
    dist_lr, dist_l_root = None, None

    if len(smoothed_points) >= 2:
        dist_lr = np.sqrt(((smoothed_points[0][0] - smoothed_points[1][0]) / w)**2 + 
                          ((smoothed_points[0][1] - smoothed_points[1][1]) / h)**2)

    if len(smoothed_points) >= 3:
        dist_l_root = np.sqrt(((smoothed_points[0][0] - smoothed_points[2][0]) / w)**2 + 
                             ((smoothed_points[0][1] - smoothed_points[2][1]) / h)**2)

    # ================= 存資料 =================
    record = {
        'left_x': smoothed_points[0][0] if len(smoothed_points) >= 1 else None,
        'left_y': smoothed_points[0][1] if len(smoothed_points) >= 1 else None,
        'right_x': smoothed_points[1][0] if len(smoothed_points) >= 2 else None,
        'right_y': smoothed_points[1][1] if len(smoothed_points) >= 2 else None,
        'root_x': smoothed_points[2][0] if len(smoothed_points) >= 3 else None,
        'root_y': smoothed_points[2][1] if len(smoothed_points) >= 3 else None,
        'distance_lr': dist_lr,
        'distance_lroot': dist_l_root,
        'confidence': obj_conf
    }

    records.append(record)

    # 每100筆存一次
    if len(records) >= 100:
        df = pd.DataFrame(records)
        df.to_excel(f"output_{save_count}.xlsx", index=False)
        print(f"已存 output_{save_count}.xlsx")
        save_count += 1
        records = []

    # 顯示畫面
    cv2.imshow("frame", img)

    # 按 q 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()