import os
import xml.dom.minidom
import cv2
import torch
from fcos.predict import prediction
from tqdm import tqdm
import fcos.get_image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import fcos.map_function as mf
from fcos.DataLoader import ValSet
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
    test_set = ValSet()
    loader = Data.DataLoader(
        dataset=test_set,  # torch TensorDataset format
        batch_size=1,  # mini batch size
        shuffle=True,
        num_workers=2,
    )
    # initialize map curve
    mAP_cup = []
    mAP_plate = []
    mAP_bowl = []
    mAP_towel = []
    mAP_shoes = []
    mAP_sponge = []
    mAP_bottle = []
    mAP_toothbrush = []
    mAP_toothpaste = []
    mAP_tray = []
    mAP_sweater = []
    mAP_cellphone = []
    mAP_banana = []
    mAP_medicine_bottle = []
    mAP_reading_glasses = []
    mAP_flashlight = []
    mAP_pill_box = []
    mAP_book = []
    mAP_knife = []
    mAP_cellphone_charger = []
    mAP_shopping_bag = []
    mAP_keyboard = []

    mAP_all = [mAP_cup, mAP_plate, mAP_bowl, mAP_towel, mAP_shoes, mAP_sponge, mAP_bottle, mAP_toothbrush,
               mAP_toothpaste, mAP_tray, mAP_sweater, mAP_cellphone, mAP_banana, mAP_medicine_bottle,
               mAP_reading_glasses, mAP_flashlight, mAP_pill_box, mAP_book, mAP_knife, mAP_cellphone_charger,
               mAP_shopping_bag,
               mAP_keyboard]
    # initialize the total of gt of each class
    gt_cup = 0
    gt_plate = 0
    gt_bowl = 0
    gt_towel = 0
    gt_shoes = 0
    gt_sponge = 0
    gt_bottle = 0
    gt_toothbrush = 0
    gt_toothpaste = 0
    gt_tray = 0
    gt_sweater = 0
    gt_cellphone = 0
    gt_banana = 0
    gt_medicine_bottle = 0
    gt_reading_glasses = 0
    gt_flashlight = 0
    gt_pill_box = 0
    gt_book = 0
    gt_knife = 0
    gt_cellphone_charger = 0
    gt_shopping_bag = 0
    gt_keyboard = 0

    gt_all = [gt_cup, gt_plate, gt_bowl, gt_towel, gt_shoes, gt_sponge, gt_bottle, gt_toothbrush, gt_toothpaste,
              gt_tray, gt_sweater, gt_cellphone, gt_banana, gt_medicine_bottle, gt_reading_glasses, gt_flashlight,
              gt_pill_box, gt_book, gt_knife, gt_cellphone_charger, gt_shopping_bag, gt_keyboard]
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
