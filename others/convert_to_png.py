from PIL import Image
import os

def convert_jpg_to_png(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all JPG files in the input folder
    jpg_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    for jpg_file in jpg_files:
        # Open the JPG image
        image = Image.open(os.path.join(input_folder, jpg_file))

        # Create the output PNG filename by replacing the extension
        png_file = os.path.splitext(jpg_file)[0] + '.png'

        # Save the image as PNG in the output folder
        image.save(os.path.join(output_folder, png_file), 'png')

        # Close the image
        image.close()

# Specify the input and output folders
input_folder = r'C:\Users\yvona\Documents\NPU_research\dataset\TITS\ALE\ALE_split\test\testA'
output_folder = r'C:\Users\yvona\Documents\NPU_research\dataset\TITS\ALE\ALE_split\test\testA_png'

# Convert JPG to PNG
convert_jpg_to_png(input_folder, output_folder)
