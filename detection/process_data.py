import os
import cv2

folder_path = "cat_dataset/images"

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        cv2.imwrite(image_path, image)

cv2.destroyAllWindows()
