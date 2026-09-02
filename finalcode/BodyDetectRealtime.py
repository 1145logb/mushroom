import cv2
import numpy as np
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. 載入模型 (路徑請確保正確)
model_path = r"C:\Users\sywan\Desktop\code\runs\segment\train9\weights\best.pt"
model = YOLO(model_path)

target_height = 1000

# 2. 開啟攝影機
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

print("開始實時偵測（主體紅色上色與累加計數）。選取影像視窗後，按下 'q' 鍵即可退出程式。")

# --- 新增：用來記錄總數，以及上一幀的數量 ---
total_count = 0 
prev_count = 0

# 3. 進入實時影像迴圈
while True:
    ret, frame = cap.read()
    if not ret:
        print("無法獲取畫面")
        break

    # 4. 對當前畫面進行推論
    results = model.predict(source=frame, conf=0.8, verbose=False)

    display_img = frame.copy()
    
    # 記錄當前這個「瞬間」畫面上的靈芝數量
    current_count = 0 

    for r in results:
        h_img, w_img = frame.shape[:2]

        # 5. 確認是否有偵測到 mask 資料
        if r.masks is not None and len(r.masks.data) > 0:
            current_count = len(r.masks.data)
            
            # 處理所有偵測到的物件 (塗成紅色)
            for mask_data in r.masks.data:
                mask = mask_data.cpu().numpy()
                mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                mask = (mask * 255).astype(np.uint8) 

                red_img = np.zeros_like(frame, dtype=np.uint8)
                red_img[:] = (0, 0, 255) 

                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                masked_red = cv2.bitwise_and(red_img, red_img, mask=mask)

                alpha = 0.6
                blended_mushroom = cv2.addWeighted(masked_frame, 1.0 - alpha, masked_red, alpha, 0)

                inverse_mask = cv2.bitwise_not(mask)
                display_img = cv2.bitwise_and(display_img, display_img, mask=inverse_mask)
                display_img = cv2.add(display_img, blended_mushroom)

    # --- 核心計數邏輯 ---
    # 如果「現在畫面上的數量」大於「上一個畫面的數量」，代表有新的靈芝進來了
    if current_count > prev_count:
        # 把多出來的數量加到總數裡
        total_count += (current_count - prev_count)
        if(total_count > 6 ):
            total_count = 0
    
    # 更新 prev_count，讓下一個迴圈可以做比較
    prev_count = current_count

    # 將「累計總數」印在畫面上 (左上角，黃色字體)
    cv2.putText(display_img, f"Total Count: {total_count}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

    # 6. 等比例縮放顯示畫面
    h_display, w_display = display_img.shape[:2]
    if h_display > 0:
        scale_ratio = target_height / h_display
        new_width = int(w_display * scale_ratio)
        resized = cv2.resize(display_img, (new_width, target_height), interpolation=cv2.INTER_AREA)
        cv2.imshow("Real-time Mushroom Coloring & Accumulating", resized)
    else:
        cv2.imshow("Real-time Mushroom Coloring & Accumulating", display_img)

    # 7. 按下 'q' 鍵退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('z'):
        total_count = 0

# 8. 釋放資源
cap.release()
cv2.destroyAllWindows()