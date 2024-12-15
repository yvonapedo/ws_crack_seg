import os
import numpy as np
from PIL import Image
import cv2


def calculate_ods(predict_folder, ground_truth_folder):
    predict_files = os.listdir(predict_folder)
    ground_truth_files = os.listdir(ground_truth_folder)

    intersection_sum = 0
    union_sum = 0

    for file in predict_files:
        if file in ground_truth_files:
            predict_path = os.path.join(predict_folder, file)
            ground_truth_path = os.path.join(ground_truth_folder, file)


            print(predict_path)
            print(ground_truth_path)
            predict_mask = cv2.imread(predict_path, cv2.IMREAD_GRAYSCALE)
            ground_truth_mask = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            print(predict_mask)
            print(ground_truth_mask)


            intersection = np.logical_and(predict_mask, ground_truth_mask)
            union = np.logical_or(predict_mask, ground_truth_mask)

            intersection_sum += np.sum(intersection)
            union_sum += np.sum(union)

    ods = intersection_sum / union_sum
    return ods

# Provide the paths to the predict and ground truth folders
predict_folder = "C:/Users/yvona/Documents/NPU_research/research/SSVS/results/crack500_128_256usseg02vit/test_2_old/predict"

ground_truth_folder = "C:/Users/yvona/Documents/NPU_research/research/SSVS/datasets/crack500_128/test_A"

ods_score = calculate_ods(predict_folder, ground_truth_folder)
print("ODS:", ods_score)
