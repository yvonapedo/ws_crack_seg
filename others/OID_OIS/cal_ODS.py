import csv
from itertools import islice
import os
from seg_for_ODS import seg_for_ods


# mask_path = 'E:\\work\\CFD-test\\masks-resize'
# save_dir =  'E:\\work\\CFD-test\\TransCNN\\epoch-2'

save_dir = 'C:\\Users\\yvona\\Documents\\NPU_research\\research\\SSVS\\attentions\\'


dir_name = save_dir.split('\\')[-1]
# result_path = os.path.join(save_dir, 'results-'+ dir_name + '-concat')

# 2 4 6 7 8 9 10 16 21
mask_path = r"C:\Users\yvona\Documents\NPU_research\dataset\TITS\ALE\ALE_split\test\testA_png"
result_path = r"C:\Users\yvona\Documents\NPU_research\dataset\TITS\ALE\ALE_split\test\result"


result_save_path = os.path.join(save_dir, 'ODS')

if not os.path.exists(result_save_path):
    os.mkdir(result_save_path)
seg_for_ods(mask_path, result_path, result_save_path)

max_f1 = 0
max_IoU = 0
max_MIoU = 0
max_Precision = 0
max_Recall = 0
max_Dice = 0
for i in range(100):
    threshold = 0.01 + i * 0.01
    name = str(threshold) + '.csv'
    total_F1 = 0
    total_IOU = 0
    total_mIOU = 0
    total_Precision = 0
    total_Recall = 0
    total_Dice = 0
    with open(os.path.join(result_save_path, name), 'r') as f:
        reader = csv.reader(f)
        for row in islice(reader, 1, None):
            print(row[1])
            total_F1 += float(row[1])
            total_IOU += float(row[2])
            total_mIOU += float(row[3])
            total_Precision += float(row[4])
            total_Recall += float(row[5])
            total_Dice += float(row[6])
    total_F1 /= 1435
    total_IOU /= 1435
    total_mIOU /= 1435
    total_Precision /= 1435
    total_Recall /= 1435
    total_Dice /= 1435
    print(threshold, total_F1, total_IOU, total_mIOU, total_Precision, total_Recall, total_Dice)
    if total_F1 > max_f1:
        max_f1 = total_F1
    if total_IOU > max_IoU:
        max_IoU = total_IOU
        max_MIoU = total_mIOU
        max_Precision = total_Precision
        max_Recall = total_Recall
        max_Dice = total_Dice

ODS = max_f1

print("===============")
print(ODS, max_IoU, max_MIoU, max_Precision, max_Recall, max_Dice)

