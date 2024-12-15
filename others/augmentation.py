import os
from PIL import Image

def augment_images(input_folder_path, output_folder_path):
    for filename in os.listdir(input_folder_path):
        with Image.open(os.path.join(input_folder_path, filename)) as im:
            # Rotate image by 90 degrees
            im_rotated = im.rotate(90)
            im_rotated.save(os.path.join(output_folder_path, 'rotated_90_' + filename))

            # Flip image horizontally
            im_flipped_h = im.transpose(Image.FLIP_LEFT_RIGHT)
            im_flipped_h.save(os.path.join(output_folder_path, 'flipped_h_' + filename))

            # Flip image vertically
            im_flipped_v = im.transpose(Image.FLIP_TOP_BOTTOM)
            im_flipped_v.save(os.path.join(output_folder_path, 'flipped_v_' + filename))

input_folder_path = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\CFD_128_128\New_128'
output_folder_path = r'C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\CFD_128_128\New_128'
augment_images(input_folder_path, output_folder_path)