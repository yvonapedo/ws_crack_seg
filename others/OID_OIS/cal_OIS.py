import csv
from itertools import islice
import os
from seg_for_OIS import seg_for_ois

# mask_path = 'E:\\work\\CFD-test\\masks-resize'
# save_dir =  'E:\\work\\CFD-test\\TransCNN\\epoch-6'

save_dir = 'C:\\Users\\yvona\\Documents\\NPU_research\\research\\SSVS\\attentions\\'
# mask_path = "C:\\Users\\yvona\\Documents\\NPU_research\\research\\SSVS\\datasets\\CFD_128_128\\test\\test_A"
# result_path = "C:\\Users\\yvona\\Documents\\NPU_research\\research\\SSVS/results/CFD_128_chAtt_plus_usseg_1/test_4\\predicts"
mask_path = r"C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\crack500_128\test_A"
result_path = r"C:\Users\yvona\Documents\NPU_research\research\SSVS\results\crack500_chanAtt\test_2\predicts"


dir_name = save_dir.split('\\')[-1]
# result_path = os.path.join(save_dir, 'results-'+ dir_name + '-concat')
result_save_path = os.path.join(save_dir, 'OIS')

if not os.path.exists(result_save_path):
    os.mkdir(result_save_path)
seg_for_ois(mask_path, result_path, result_save_path)
total_F1 = 0

for res in os.listdir(result_path):
    print("=================res")

    name = res.split('.')[0]+".csv"
    print(name)

# for i in range(1, 541):
#     name = str(i) + '.csv'
#     print("=====-----name-----=====")
#     print(name)

    max_f1 = 0
    with open(os.path.join(result_save_path, name), 'r') as f:
        reader = csv.reader(f)
        for row in islice(reader, 1, None):
            print(row[1])
            if float(row[1]) > max_f1:
                max_f1 = float(row[1])
    print(name, max_f1)
    total_F1 += max_f1

OIS = total_F1 / 1435
print("===================")
print(OIS)