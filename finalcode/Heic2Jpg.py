from PIL import Image
import pillow_heif
import os

# 註冊 HEIC 支援
pillow_heif.register_heif_opener()

# HEIC 資料夾
input_folder = r"D:\gldownloads\靈芝照片0517-20260815T081744Z-1-001\靈芝照片0517\醜的-藍色背景"

# JPG 輸出資料夾
output_folder = r"C:\Users\sywan\Desktop\code\mushP0517\normal\blue"

os.makedirs(output_folder, exist_ok=True)

# 批量轉換
for filename in os.listdir(input_folder):

    if filename.lower().endswith(".heic"):

        heic_path = os.path.join(input_folder, filename)

        jpg_name = os.path.splitext(filename)[0] + ".jpg"

        jpg_path = os.path.join(output_folder, jpg_name)

        image = Image.open(heic_path)

        image.convert("RGB").save(jpg_path, "JPEG", quality=95)

        #print(f"已轉換: {jpg_name}")

print("全部完成")