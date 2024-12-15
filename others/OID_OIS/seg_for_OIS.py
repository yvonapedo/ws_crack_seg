import os
from PIL import Image
import numpy as np
import csv
import cv2


"""
计算分割的性能

TP (True Positive)：真正例，即模型预测为正例，实际也为正例；
FP (False Positive)：假正例，即模型预测为正例，实际为反例；
FN (False Negative)：假反例，即模型预测为反例，实际为正例；
TN (True Negative)：真反例，即模型预测为反例，实际也为反例。
将这四种指标放在同一表格中，便构成了混淆矩阵(横着代表预测值，竖着代表真实值):
P\L     预测P    预测N
真实P      TP      FP
真实N      FN      TN
"""


# 计算混淆矩阵
def get_hist(label_true, label_pred, n_class):
    
    # label_true是转化为一维数组的真实标签，label_pred是转化为一维数组的预测结果，n_class是类别数
    # hist是一个混淆矩阵(一个二维数组)，可以写成hist[label_true][label_pred]的形式
    
    # mask在和label_true相对应的索引的位置上填入true或者false
    # label_true[mask]会把mask中索引为true的元素输出
    mask = (label_true >= 0) & (label_true < n_class)
    # n_class * label_true[mask].astype(int) + label_pred[mask]计算得到的是二维数组元素变成一位数组元素的时候的地址取值(每个元素大小为1)，返回的是一个numpy的list
    # np.bincount()会给出索引对应的元素个数
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2)
    hist = hist.reshape(n_class, n_class)
    
    return hist


# precision（精确率）
def cal_Precision(hist):
    # precision = TP / TP + FP
    if hist[1][1] == 0:
        return 0
    precision = hist[1][1] / (hist[0][1] + hist[1][1])
    return precision


# recall（召回率）
def cal_Recall(hist):
    # recall = TP / TP + FN
    if hist[1][1] == 0:
        return 0
    recall = hist[1][1] / (hist[1][0] + hist[1][1])
    return recall


def cal_F1(precision, recall):
    if precision == 0 or recall == 0:
        return 0
    F1 = 2 * precision * recall / (precision + recall)
    return F1


def cal_IOU(hist):
    if hist[1][1] == 0:
        return 0
    IOU = hist[1][1] / (hist[0][1] + hist[1][0] + hist[1][1])
    return IOU


def cal_mIOU(hist):
    if hist[0][0] == 0 and hist[1][1] == 0:
        return 0
    elif hist[0][0] == 0:
        IOU1 = 0
        IOU2 = hist[1][1] / (hist[0][1] + hist[1][0] + hist[1][1])
        mIOU = (IOU1 + IOU2) / 2
    elif hist[1][1] == 0:
        IOU1 = hist[0][0] / (hist[0][0] + hist[0][1] + hist[1][0])
        IOU2 = 0
        mIOU = (IOU1 + IOU2) / 2
    else:
        IOU1 = hist[0][0] / (hist[0][0] + hist[0][1] + hist[1][0])
        IOU2 = hist[1][1] / (hist[0][1] + hist[1][0] + hist[1][1])
        mIOU = (IOU1 + IOU2) / 2
    return mIOU


def seg(norm, mask, k, num, result_save_path):
    norm = norm / 255.
    # k is the threshold
    norm[norm < k] = int(0)
    norm[norm >= k] = int(1)
    mask = mask.astype('int64')
    norm = norm.astype('int64')
    hist = get_hist(mask, norm, 2)
    p = cal_Precision(hist)
    r = cal_Recall(hist)
    f = cal_F1(p, r)
    print("Threshold: {:.2f}, F1-score: {:.4f}".format(k, f))
    # 保存测试数据到csv文件
    item = {'Threshold': str(k), 'F1-score': str(f)}
    fieldnames = ['Threshold', 'F1-score']
    save_result = os.path.join(result_save_path, str(num) + '.csv')
    with open(save_result, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 判断表格内容是否为空
        if not os.path.getsize(save_result):
            writer.writeheader()  # 写入表头
        writer.writerows([item])
    return

def seg_for_ois(mask_path, res_path, result_save_path):
    for res in os.listdir(res_path):
        print("=================res")
        print(res)
        res_img = np.array(Image.open(os.path.join(res_path, res)).convert('L'))
        mask = np.array(Image.open(os.path.join(mask_path, res)).convert('1'))
        number = res.split('.')[0]
        for i in range(100):
            threshold = 0.01 + i * 0.01
            seg(res_img, mask, threshold, number, result_save_path)