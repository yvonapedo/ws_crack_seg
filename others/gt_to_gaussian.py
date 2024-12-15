import os
import cv2

# Define paths to input and output folders
input_folder = r"C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\part_crack500\trainA"
output_folder = r"C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\part_crack500\gaussian"

# Define kernel size and standard deviation of Gaussian filter
ksize = (5, 5) # kernel size
sigma = 0.5    # standard deviation

# Iterate over each file in the input folder
for filename in os.listdir(input_folder):
    # Load image
    img = cv2.imread(os.path.join(input_folder, filename))
    # Apply Gaussian filter
    img_filtered = cv2.GaussianBlur(img, ksize, sigma)
    # Save filtered image to output folder with the same filename
    cv2.imwrite(os.path.join(output_folder, filename), img_filtered)
