import os
import numpy as np
import cv2

def calculate_ois(predict_folder, ground_truth_folder):
    predict_files = os.listdir(predict_folder)
    ground_truth_files = os.listdir(ground_truth_folder)

    iou_scores = []


    for file in predict_files:
        if file in ground_truth_files:
            predict_path = os.path.join(predict_folder, file)
            ground_truth_path = os.path.join(ground_truth_folder, file)

            predict_mask = cv2.imread(predict_path, cv2.IMREAD_GRAYSCALE)
            ground_truth_mask = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

            intersection = np.logical_and(predict_mask, ground_truth_mask)
            union = np.logical_or(predict_mask, ground_truth_mask)

            iou = np.sum(intersection) / np.sum(union)
            iou_scores.append(iou)

    ois = np.mean(iou_scores)
    return ois

# Provide the paths to the predict and ground truth folders
predict_folder = "C:/Users/yvona/Documents/NPU_research/research/SSVS/results/crack500_128_256usseg02vit/test_2_old/predict"
ground_truth_folder = "C:/Users/yvona/Documents/NPU_research/research/SSVS/datasets/crack500_128/test_A"

ois_score = calculate_ois(predict_folder, ground_truth_folder)
print("OIS:", ois_score)
# OIS: 0.4160678847677176
# ODS: 0.4134926545703797