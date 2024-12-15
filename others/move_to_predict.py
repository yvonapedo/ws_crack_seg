import os
import shutil
import cv2

import shutil

def copy_hop_files(src_folder, dst_folder):
    for filename in os.listdir(src_folder):
        if filename.endswith("_fake_A.png"):
            print(filename)
            src_path = os.path.join(src_folder, filename)
            new_filename = filename[:-11] + ".png"

            dst_path = os.path.join(dst_folder, new_filename)
            shutil.copy2(src_path, dst_path)

copy_hop_files(r"C:\Users\yvona\Documents\NPU_research\research\SSVS\results\ssvs_chAtt_usseg_1\test_38\images",
               r"C:\Users\yvona\Documents\NPU_research\research\SSVS\results\ssvs_chAtt_usseg_1\test_38\predict")


