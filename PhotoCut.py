import cv2
import os

input_folder = (r'C:\Users\User\Desktop\deeplearning\mushroomphoto\pretty_jpg')
output_folder = (r"C:\Users\User\Desktop\mushrrom_p\pretty")
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)

    # 轉為灰階、二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 找輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大輪廓（假設靈芝是圖中的主體）
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 裁切
        cropped = image[y:y+h, x:x+w]
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, cropped)

print("依靈芝位置裁切完成 ✅")
