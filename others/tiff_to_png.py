import os
from PIL import Image

def convert_tiff_to_png(input_folder_path, output_folder_path):
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.tiff') or filename.endswith('.tif'):
            with Image.open(os.path.join(input_folder_path, filename)) as im:
                im.save(os.path.join(output_folder_path, os.path.splitext(filename)[0] + '.png'))

input_folder_path = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\CFD_128_128\New_128'
output_folder_path = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\CFD_128_128\New_128\png'
convert_tiff_to_png(input_folder_path, output_folder_path)