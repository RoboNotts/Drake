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
from DataLoader import FolderData
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
    num = 0  # the total number of ground truth
    detected_gt = []
    for box in boxes:
        TP = False  # judge if the bndbox is True Positive
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


def returnMAP(net):
    net.cuda()
    net.eval()
    # load test set
    test_set = FolderData("./DataSet/labels/val/")
    loader = Data.DataLoader(
        dataset=test_set,  # torch TensorDataset format
        batch_size=1,  # mini batch size
        shuffle=True,
        num_workers=2,
    )
    
    # load class list
    try:
        with open('./classes.txt', 'r') as f:
            # obtain class list
            label_list = f.read().splitlines()
    except FileNotFoundError:
        print("classes.txt file was not found...")
        exit(0)  
    
    mAP_all = []
    gt_all  = []
    for i in classes:
        mAP_all.append([])
        gt_all.append(0)
    
    # traverse validation set
    for step, label_paths in tqdm(enumerate(loader)):
        # get an image
        torch_images, labels = get_image.get_label(label_paths)
        labels = labels[0]
        torch_images = torch_images.cuda()
        # predict
        row = torch_images.shape[2]
        col = torch_images.shape[3]
        confs, locs, centers = net(torch_images)
        boxes = prediction(confs, locs, centers, row, col)
        boxes.sort(key=take2, reverse=True)
        for i in range(len(gt_all)):
            gt_all[i] += set_mAP_points(boxes, labels, mAP_all[i], i)
    mAP = 0
    for i in range(len(mAP_all)):
        mAP_all[i].sort(key=take2, reverse=True)
        p, r, ap = compute_mAP(mAP_all[i], gt_all[i])
        mAP += ap * gt_all[i]
    mAP = mAP / sum(gt_all)
    return mAP
