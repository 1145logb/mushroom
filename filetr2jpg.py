import os
from PIL import Image
import pillow_heif

# 開啟 pillow-heif 支援
pillow_heif.register_heif_opener()

def convert_heic_to_jpg(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".heic"):
            heic_path = os.path.join(input_folder, filename)
            jpg_filename = os.path.splitext(filename)[0] + ".jpg"
            jpg_path = os.path.join(output_folder, jpg_filename)

            try:
                image = Image.open(heic_path)
                image = image.convert("RGB")  # 確保為 RGB 模式
                image.save(jpg_path, "JPEG")
                print(f"Converted: {filename} → {jpg_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

# 使用範例
input_folder = r"C:\Users\User\Desktop\deeplearning\mushroomphoto\seperate"
output_folder = r"C:\Users\User\Desktop\deeplearning\mushroomphoto\seperate_jpg"

convert_heic_to_jpg(input_folder, output_folder)
