import cv2
import numpy as np

#resize image
def resize_and_show(image, window_name='Resized Image'):
   
    new_width = 500  
    new_height =  800 

   
    resized_image = cv2.resize(image, (new_width, new_height))

    
    cv2.imshow(window_name, resized_image)    
def photo_to_ycrcb(photo_path):
    image = cv2.imread(photo_path)
    if image is None:
        print("Error: Could not read the image.")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    image_y = image_ycrcb[:, :, 0]  
    img_edge = cv2.medianBlur(image_y, 17)  
    img_edge = cv2.Laplacian(image_y, -1, 11, 5)  
    resize_and_show(image, 'Original Image')
    resize_and_show(gray, 'Gray Image')
    resize_and_show(binary, 'Binary Image')
    resize_and_show(image_hsv, 'HSV Image')
    resize_and_show(image_ycrcb, 'YCrCb Image')
    resize_and_show(image_y, 'Y Channel Image')
    resize_and_show(img_edge, 'Edge Detected Image')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    photo_path = r"C:\Users\User\Desktop\deeplearning\mushroomphoto\pretty_jpg\IMG_0634.jpg" 
    photo_to_ycrcb(photo_path)
