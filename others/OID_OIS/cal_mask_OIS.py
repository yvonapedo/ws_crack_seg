import csv
from itertools import islice
import os
from seg_for_OIS import seg_for_ois

# input_output_path = r'C:\Users\yvona\Documents\NPU_research\research\111\111'
# save_dir = r'C:\Users\yvona\Documents\NPU_research\research\111\111'

save_dir ='C:\\Users\\yvona\\Documents\\NPU_research\\research\\SSVS\\attentions\\'
mask_path = r"C:\Users\yvona\Documents\NPU_research\research\SSVS\datasets\crack500_128"
result_path = r"C:\Users\yvona\Documents\NPU_research\research\SSVS\results\crack500_chanAtt\test_2"


def cal_mask_ois(mask_path, save_dir, result_path):
    mask_path = mask_path + '/test_A'
    mask_result_path = os.path.join(result_path, 'predicts')
    result_save_path = os.path.join(save_dir, 'mask_OIS')
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    seg_for_ois(mask_path, mask_result_path, result_save_path)

    total_F1 = 0
    for i in range(1, len(os.listdir(mask_path)) + 1):

        name = str(i) + '.csv'
        # name = os.path.join(result_save_path, name)
        print(name)
        print(result_save_path)
        # if not os.path.exists(name):
        #     os.mkdir(name)

        max_f1 = 0
        with open(os.path.join(result_save_path, name), 'r') as f:
            reader = csv.reader(f)
            for row in islice(reader, 1, None):
                # print(row[1])
                if float(row[1]) > max_f1:
                    max_f1 = float(row[1])
        # print(name, max_f1)
        total_F1 += max_f1

    print("OIS: {}".format(total_F1 / len(os.listdir(mask_path))))

cal_mask_ois(mask_path, save_dir, result_path)