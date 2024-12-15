import os
from PIL import Image

# set source and destination directories
src_dir = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\special\testA_128'
dest_dir = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\special\testA_128\good'

# loop through all images in the source directory
for filename in os.listdir(src_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # open the image and check if it contains any white pixels
        img = Image.open(os.path.join(src_dir, filename))
        if (255, 255, 255) not in img.getdata():
            # if the image doesn't contain any white pixels, move it to the destination directory
            os.rename(os.path.join(src_dir, filename), os.path.join(dest_dir, filename))
