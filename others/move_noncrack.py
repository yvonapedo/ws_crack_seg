import os
import shutil

# set the source and destination directories
src_dir = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\special\testB_128'
dest_dir = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\special\testB_128\good'

# read the image filenames from the test.txt file
with open('test.txt', 'r') as f:
    image_list = f.read().splitlines()

# loop through the image filenames in the list and move them to the destination directory
for filename in image_list:
    if filename.endswith('.jpg') or filename.endswith('.png'):
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        shutil.move(src_path, dest_path)
