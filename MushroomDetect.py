from ultralytics import YOLO
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# === 設定路徑 ===
model_path = r'C:\Users\User\Desktop\deeplearning\runs\detect\train8\weights\best.pt'  # 訓練好的模型
input_folder = r'C:\Users\User\Desktop\deeplearning\mushroomphoto\pretty_jpg'                   # 要預測的圖片資料夾
output_folder = r'C:\Users\User\Desktop\mushrrom_p\pretty'               # 輸出結果資料夾
os.makedirs(output_folder, exist_ok=True)

# === 載入模型 ===
model = YOLO(model_path)

# === 預測每一張圖片 ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        results = model(img_path)[0]  # 預測

        image = cv2.imread(img_path)
        for box in results.boxes:
            # 抓出框的左上與右下座標
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            # 畫框與標籤
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 儲存結果圖
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, image)

print("✅ 所有圖片已完成預測與畫框儲存！")
