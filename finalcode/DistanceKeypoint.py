from ultralytics import YOLO
import cv2
import os
import pandas as pd
import numpy as np

# --- 解決 OpenMP 錯誤 ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# === 設定模型與路徑 ===
# 只保留 Keypoint 模型
model_keypoint_path = r"C:\Users\sywan\Desktop\code\runs\segment\train9\weights\best.pt"
input_dir = r"C:\Users\sywan\Desktop\code\mushP"
output_dir = r"C:\Users\sywan\Desktop\code\mushPing"
output_excel = r"C:\Users\sywan\Desktop\code\mushbody.xlsx"

# 建立輸出資料夾
os.makedirs(output_dir, exist_ok=True)

# 載入 Keypoint 模型
model_keypoint = YOLO(model_keypoint_path)

# Keypoint 名稱
keypoint_names = ['left', 'right', 'root']

# 儲存所有結果的列表
records = []

# 處理資料夾中的所有圖片
for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_dir, filename)
    img = cv2.imread(image_path)
    if img is None:
        continue
    
    h, w = img.shape[:2]

    # --- 直接使用關鍵點偵測模型對全圖進行預測 ---
    # 設定 conf 門檻，並關閉 verbose 減少輸出
    results = model_keypoint.predict(source=img, conf=0.25, verbose=False)

    # 檢查是否有偵測到任何目標與關鍵點
    if (not hasattr(results[0], "keypoints") or 
        results[0].keypoints is None or 
        len(results[0].keypoints.xy) == 0):
        print(f"⚠️ 無法在圖片中偵測到關鍵點：{filename}")
        continue

    # 取得信心值最高的目標 (索引 0)
    # xy 座標已經是相對於原圖的像素值
    keypoints_xy = results[0].keypoints.xy[0].cpu().numpy()
    keypoints_conf = (
        results[0].keypoints.conf[0].cpu().numpy()
        if results[0].keypoints.conf is not None
        else np.ones(len(keypoints_xy))
    )
    
    # 取得該目標的整體信心值 (Bounding Box Confidence)
    obj_conf = float(results[0].boxes.conf[0]) if results[0].boxes is not None else 0.0

    # 畫出關鍵點
    for idx, ((kx, ky), kv) in enumerate(zip(keypoints_xy, keypoints_conf)):
        # 信心值大於 0.5 畫綠色，否則畫灰色
        color = (0, 255, 0) if kv > 0.5 else (100, 100, 100)
        cv2.circle(img, (int(kx), int(ky)), 10, color, -1)
        
        label = keypoint_names[idx] if idx < len(keypoint_names) else f"k{idx}"
        cv2.putText(img, f"{label} ({kv:.2f})", (int(kx)+10, int(ky)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # --- 計算距離 (改用像素距離或相對於全圖的比例) ---
    dist_lr = None
    dist_l_root = None
    
    if len(keypoints_xy) >= 2:
        # 計算 Left 到 Right 的像素距離，並除以圖片寬度歸一化
        dist_lr = np.sqrt(((keypoints_xy[0][0] - keypoints_xy[1][0]) / w)**2 + 
                          ((keypoints_xy[0][1] - keypoints_xy[1][1]) / h)**2)
    if len(keypoints_xy) >= 3:
        # 計算 Left 到 Root 的歸一化距離
        dist_l_root = np.sqrt(((keypoints_xy[0][0] - keypoints_xy[2][0]) / w)**2 + 
                             ((keypoints_xy[0][1] - keypoints_xy[2][1]) / h)**2)

    # 儲存資訊
    record = {
        'filename': filename,
        'left_x': keypoints_xy[0][0] if len(keypoints_xy) >= 1 else None,
        'left_y': keypoints_xy[0][1] if len(keypoints_xy) >= 1 else None,
        'right_x': keypoints_xy[1][0] if len(keypoints_xy) >= 2 else None,
        'right_y': keypoints_xy[1][1] if len(keypoints_xy) >= 2 else None,
        'root_x': keypoints_xy[2][0] if len(keypoints_xy) >= 3 else None,
        'root_y': keypoints_xy[2][1] if len(keypoints_xy) >= 3 else None,
        'left_visible': bool(keypoints_conf[0] > 0.5) if len(keypoints_conf) >= 1 else False,
        'right_visible': bool(keypoints_conf[1] > 0.5) if len(keypoints_conf) >= 2 else False,
        'root_visible': bool(keypoints_conf[2] > 0.5) if len(keypoints_conf) >= 3 else False,
        'distance_left_right_norm': dist_lr,
        'distance_left_root_norm': dist_l_root,
        'obj_confidence': obj_conf
    }
    records.append(record)

    # 儲存標記後影像
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, img)

# 輸出到 Excel
if records:
    df = pd.DataFrame(records)
    df.to_excel(output_excel, index=False)
    print(f"\n✅ 處理完成！共分析 {len(records)} 張圖片。")
    print(f"📊 Excel 檔案已儲存至：{output_excel}")
else:
    print("\n⚠️ 偵測失敗，請檢查模型路徑或圖片內容。")