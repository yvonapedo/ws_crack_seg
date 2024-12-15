from PIL import Image
import os

# Specify the input and output directories
input_directory = 'C:/Users/yvona/Documents/NPU_research/research/SSVS/datasets/metu/test'
output_directory = 'C:/Users/yvona/Documents/NPU_research/research/SSVS/datasets/metu/test1'

from PIL import Image
import os

# Specify the input and output directories
# input_directory = 'input_folder'
# output_directory = 'output_folder'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# List all files in the input directory
file_list = os.listdir(input_directory)

for filename in file_list:
    # Check if the file is an image (you can add more image formats as needed)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        # Open the image
        image_path = os.path.join(input_directory, filename)
        img = Image.open(image_path)

        # Convert the image to black & white (binary)
        img = img.convert('1')  # '1' mode converts to 1-bit pixels (black and white)

        # Save the processed image to the output directory
        output_path = os.path.join(output_directory, filename)
        img.save(output_path)

print("Conversion complete. Black & white images saved in the output directory.")
