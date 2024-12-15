from PIL import Image
import numpy as np

# Open an image using Pillow
# image = Image.open(r'C:\Users\yvona\Documents\NPU_research\research\SSVS\results\metu_usseg02vitcls\test_2\predict\07045.png')

import cv2
import numpy as np

# Load the image using OpenCV
image = cv2.imread(r'C:\Users\yvona\Documents\NPU_research\research\SSVS\results\metu_usseg02vitcls\test_2\predict_0\07045.png', cv2.IMREAD_GRAYSCALE)  # Change 'your_image.jpg' to your image file

# Convert the image to a NumPy array
image_array = np.array(image)

# Flatten the 2D array into a 1D array
flat_array = image_array.flatten()

# Save the 1D array to a text file
np.savetxt('image_data.txt', flat_array, fmt='%d')

print("Image data has been saved to 'image_data.txt'")
