import os
import xml.dom.minidom
import cv2
import torch
from predict import prediction
from tqdm import tqdm
import get_image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import map_function as mf
from DataLoader import TestSet
import torch.utils.data as Data


def take2(elem):
    return elem[1]


def compute_mAP(points, GT_num):
    boxes_num = 0
    TP_num = 0
    AP = 0
    precision = 0
    recall = 0
    for point in points:
        boxes_num += 1
        if point[2] == True:
            TP_num += 1
            precision = TP_num / boxes_num
            AP += (1 / GT_num) * precision
        else:
            precision = TP_num / boxes_num
        recall = TP_num / GT_num
    return precision, recall, AP


def printPR(points, GT_num):
    boxes_num = 0
    TP_num = 0
    pre_points = []
    recall_points = []
    for point in points:
        boxes_num += 1
        if point[2] == True:
            TP_num += 1
            pre_points.append(TP_num / boxes_num)
        else:
            pre_points.append(TP_num / boxes_num)
        recall_points.append(TP_num / GT_num)
    plt.plot(recall_points, pre_points)
    plt.title('line chart')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    precision = np.array(pre_points)
    recall = np.array(recall_points)
    return precision, recall


def set_mAP_points(boxes, gt_boxes, mAP_boxes, tag, threshold=0.5):
    '''
	计算召回率
	参数：
		boxes：神经网络输出的物体框
		gt_boxes：标签框
		tag：被计算召回率的类别
		threshold：交并比阈值
	'''

    num = 0  # gt框总数
    detected_gt = []
    for box in boxes:
        TP = False  # 指示该预测框是否为TP
        iou_num = 0  # 统计该框与GT的IOU大于阈值的个数
        if box[0] == tag:
            for gt in gt_boxes:
                if gt[0] == tag and gt not in detected_gt:
                    iou = mf.compute_iou([gt[1], gt[2], gt[3], gt[4]], [box[2], box[3], box[4], box[5]])
                    if iou >= threshold:
                        detected_gt.append(gt)
                        TP = True
                        break
            # if iou >= threshold:
            # iou_num += 1
            # if iou_num == 1:
            # detected_gt.append(gt)
            # TP = True
            mAP_boxes.append([box[0], box[1], TP])

    for gt in gt_boxes:
        if gt[0] == tag:
            num += 1

    return num


if __name__ == '__main__':
    # 加载神经网络
    net = torch.load('./models/net50.pkl')
    net.cuda()
    net.eval()
    # load test set
    test_set = TestSet()
    loader = Data.DataLoader(
        dataset=test_set,  # torch TensorDataset format
        batch_size=1,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )
    # 定义精度和召回率
    mAP_bottle = []
    mAP_cup = []
    mAP_bowl = []
    mAP_plate = []
    # initialize the total of gt of each class
    gt_bottle = 0
    gt_cup = 0
    gt_bowl = 0
    gt_plate = 0
    # 逐图检测
    for step, label_paths in tqdm(enumerate(loader)):
        # 读取一帧
        torch_images, labels = get_image.get_label(label_paths, True)
        labels = labels[0]
        torch_images = torch_images.cuda()
        # 预测
        confs, locs, centers = net(torch_images)
        boxes = prediction(confs, locs, centers, 300, 300)
        boxes.sort(key=take2, reverse=True)
        gt_bottle += set_mAP_points(boxes, labels, mAP_bottle, 0)
        gt_cup += set_mAP_points(boxes, labels, mAP_cup, 1)
        gt_bowl += set_mAP_points(boxes, labels, mAP_bowl, 2)
        gt_plate += set_mAP_points(boxes, labels, mAP_plate, 3)
    mAP_bottle.sort(key=take2, reverse=True)
    mAP_cup.sort(key=take2, reverse=True)
    mAP_bowl.sort(key=take2, reverse=True)
    mAP_plate.sort(key=take2, reverse=True)
    p_bottle, r_bottle, ap_bottle = compute_mAP(mAP_bottle, gt_bottle)
    p_cup, r_cup, ap_cup = compute_mAP(mAP_cup, gt_cup)
    p_bowl, r_bowl, ap_bowl = compute_mAP(mAP_bowl, gt_bowl)
    p_plate, r_plate, ap_plate = compute_mAP(mAP_plate, gt_plate)
    GT = gt_bottle + gt_cup + gt_bowl + gt_plate
    mAP = (ap_bottle * gt_bottle + ap_cup * gt_cup + ap_bowl * gt_bowl + ap_plate * gt_plate) / GT
    all_boxes = mAP_bottle + mAP_cup + mAP_bowl + mAP_plate
    print('对bottle的样本：')
    print('精度为：' + str(p_bottle))
    print('召回率为：' + str(r_bottle))
    print('AP为：' + str(ap_bottle))

    print('对cup的样本：')
    print('精度为：' + str(p_cup))
    print('召回率为：' + str(r_cup))
    print('AP为：' + str(ap_cup))

    print('对bowl的样本：')
    print('精度为：' + str(p_bowl))
    print('召回率为：' + str(r_bowl))
    print('AP为：' + str(ap_bowl))

    print('对plate的样本：')
    print('精度为：' + str(p_plate))
    print('召回率为：' + str(r_plate))
    print('AP为：' + str(ap_plate))

    avg_p = (p_bottle * gt_bottle + p_cup * gt_cup + p_bowl * gt_bowl + p_plate * gt_plate) / GT
    avg_r = (r_bottle * gt_bottle + r_cup * gt_cup + r_bowl * gt_bowl + r_plate * gt_plate) / GT
    avg_ap = (ap_bottle * gt_bottle + ap_cup * gt_cup + ap_bowl * gt_bowl + ap_plate * gt_plate) / GT
    f1_score = 2 * avg_p * avg_r / (avg_p + avg_r)
    print('on average：')
    print('精度为：' + str(avg_p))
    print('召回率为：' + str(avg_r))
    print('mAP为：' + str(mAP))
    print('f1 score: ' + str(f1_score))

    all_boxes.sort(key=take2, reverse=True)
    pre, recall = printPR(all_boxes, GT)
    # np.save("./PRcurve/precision", pre)
    # np.save("./PRcurve/recall", recall)
