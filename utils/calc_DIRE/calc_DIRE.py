# calc_DIRE.py

import os
import numpy as np
from PIL import Image
import sys

def calculate_image_means(image_path):
    
    
    img_rgb = Image.open(image_path).convert('RGB')
    img_rgb_array = np.array(img_rgb)
    rgb_mean = np.mean(img_rgb_array)


    img_gray = Image.open(image_path).convert('L')  
    img_gray_array = np.array(img_gray)
    gray_mean = np.mean(img_gray_array)

    return rgb_mean, gray_mean

def calculate_folder_means(folder_path):
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]

    if not image_files:
        print(f"{os.path.basename(folder_path)}: No images found")
        return


    rgb_means = []
    gray_means = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        rgb_mean, gray_mean = calculate_image_means(image_path)
        rgb_means.append(rgb_mean)
        gray_means.append(gray_mean)

    
    folder_rgb_mean = np.mean(rgb_means)
    folder_gray_mean = np.mean(gray_means)

    
    folder_name = os.path.basename(folder_path)
    print(f"{folder_name} (RGB mean): {folder_rgb_mean}")
    print(f"{folder_name} (Grayscale mean): {folder_gray_mean}")

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python calc_DIRE.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    calculate_folder_means(folder_path)

