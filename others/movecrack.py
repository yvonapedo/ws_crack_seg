import os
import shutil

# Specify the source folder and destination folder paths
source_folder = r"C:\Users\yvona\Documents\NPU_research\research\SSVS\results\CFD_128_256_000_usseg\125__test_2\images"
destination_folder = r"C:\Users\yvona\Documents\NPU_research\research\SSVS\results\CFD_128_256_000_usseg\125__test_2\fake"

# Ensure the destination folder exists, or create it
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# List all files in the source folder
files = os.listdir(source_folder)

# Loop through the files and move those with "fake" in their names
for filename in files:
    if "fake" in filename:
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.move(source_path, destination_path)
        print(f"Moved {filename} to {destination_folder}")

print("Image movement complete.")
