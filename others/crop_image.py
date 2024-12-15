import os
import cv2

# Define paths to input and output folders
input_folder = r"C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\special\CRACK500\testB_org"
output_folder = r"C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\special\CRACK500\testB"

# Define crop size
crop_size = 128

# Iterate over each file in the input folder
for filename in os.listdir(input_folder):
    # Load image
    img = cv2.imread(os.path.join(input_folder, filename))
    # Get image dimensions
    height, width = img.shape[:2]
    # Calculate number of crops in the x and y directions
    num_crops_x = width // crop_size
    num_crops_y = height // crop_size
    # Crop images and save them to output folder
    for i in range(num_crops_x):
        for j in range(num_crops_y):
            x = i * crop_size
            y = j * crop_size
            crop_img = img[y:y+crop_size, x:x+crop_size]
            # print("----")
            crop_filename = os.path.splitext(filename)[0] + f"_crop_{i}_{j}.png"
            cv2.imwrite(os.path.join(output_folder, crop_filename), crop_img)
