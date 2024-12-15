import cv2
import os

# Define paths for input and output folders
input_folder = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\crackTree_128\trainC_256'
output_folder = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\crackTree_128\trainC_512'

# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    # Read image from file
    img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)

    # Flip image vertically and concatenate to create 256x128 image
    img_vflip = cv2.flip(img, 0)
    img_concat = cv2.vconcat([img, img_vflip])

    # Flip concatenated image horizontally and create final 256x256 image
    img_hflip = cv2.flip(img_concat, 1)
    img_final = cv2.hconcat([img_concat, img_hflip])

    # Save final image to file
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, img_final)
