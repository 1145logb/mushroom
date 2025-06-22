from ultralytics import YOLO
import os
import cv2

# 载入 YOLOv8 模型
model = YOLO("yolov8n.pt")  # 你可以替换成自己的模型

# 定义图片路径
male_folder = "C:/Users/User/Desktop/deeplearning/phototest/male/"
female_folder = "C:/Users/User/Desktop/deeplearning/phototest/female/"

# 获取所有图片路径
male_images = [os.path.join(male_folder, img) for img in os.listdir(male_folder) if img.endswith((".jpg", ".png"))]
female_images = [os.path.join(female_folder, img) for img in os.listdir(female_folder) if img.endswith((".jpg", ".png"))]

# 定义保存结果的路径
output_male = "C:/Users/User/Desktop/deeplearning/results/male_results/"
output_female = "C:/Users/User/Desktop/deeplearning/results/female_results/"
os.makedirs(output_male, exist_ok=True)
os.makedirs(output_female, exist_ok=True)

# 处理男性图片
print("Processing male images...")
for idx, img_path in enumerate(male_images):
    print(f"Processing {idx+1}/{len(male_images)}: {img_path}")
    results = model(img_path)  # 运行 YOLO 进行推理
    for result in results:
        # 获取带标注的图像
        img_result = result.plot()
        save_path = os.path.join(output_male, f"male_result_{idx+1}.jpg")
        cv2.imwrite(save_path, img_result)  # 保存图片
        print(f"Saved: {save_path}")

# 处理女性图片
print("Processing female images...")
for idx, img_path in enumerate(female_images):
    print(f"Processing {idx+1}/{len(female_images)}: {img_path}")
    results = model(img_path)  # 运行 YOLO 进行推理
    for result in results:
        # 获取带标注的图像
        img_result = result.plot()
        save_path = os.path.join(output_female, f"female_result_{idx+1}.jpg")
        cv2.imwrite(save_path, img_result)  # 保存图片
        print(f"Saved: {save_path}")

print("✅ Inference completed! Check results folder.")

