# import os
# from PIL import Image
#
# def resize_images(input_folder_path, output_folder_path):
#     for filename in os.listdir(input_folder_path):
#         with Image.open(os.path.join(input_folder_path, filename)) as im:
#             if im.size == (256, 256):
#                 im_resized = im.resize((128, 128))
#                 im_resized.save(os.path.join(output_folder_path, filename))
#
# input_folder_path = r'C:\Users\yvona\Documents\NPU_research\dataset\CFD\New_256'
# output_folder_path = r'C:\Users\yvona\Documents\NPU_research\dataset\CFD\New_128'
# resize_images(input_folder_path, output_folder_path)

# import os
# import cv2
# import shutil
#
# # Set the paths of the source and destination folders
# src_folder = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\crackTree_128\trainC'
# dst_folder = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\crackTree_128\trainC_256'
#
# # Set the new size for the images
# new_size = (256, 256)
#
# # Loop through all files in the source folder
# for filename in os.listdir(src_folder):
#     # Check if the file is an image (you can use other conditions based on your needs)
#     if filename.endswith('.png') or filename.endswith('.jpg'):
#         # Load the image
#         img = cv2.imread(os.path.join(src_folder, filename))
#         # Resize the image
#         resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_NEAREST)
#         # Save the resized image to the destination folder
#         cv2.imwrite(os.path.join(dst_folder, filename), resized_img)
#         # Move the original image to a backup folder (optional)
#         shutil.move(os.path.join(src_folder, filename), os.path.join(src_folder, 'backup', filename))


import os
import cv2

# Input and output paths
input_folder = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\results\\crack500_chanAtt\test_7\predict'
output_folder = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\results\\crack500_chanAtt\test_7\predicts'

# Loop through all images in input folder
for file_name in os.listdir(input_folder):
    # Read input image
    input_path = os.path.join(input_folder, file_name)
    img = cv2.imread(input_path)

    # Perform bicubic interpolation
    resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)

    # Save output image
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, resized)
