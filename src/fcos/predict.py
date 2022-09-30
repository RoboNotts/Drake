import os
import torch
import numpy as np
import cv2
import map_function as mf
import copy
from DataLoader import TestSet, TrainSet
import torch.utils.data as Data
import get_image


def prediction(confs, locs, centers, row, col):
    lime = 0.5  # 交并比阈值
    iou_lime = 0.5  # 交并比阈值
    cls_lime = 0.2  # 置信度阈值
    # 获取各特征图的尺寸
    map_sizes = []
    for map_num in range(len(confs)):
        # 获取特征图尺寸
        H = confs[map_num].size(2)
        W = confs[map_num].size(3)
        map_sizes.append([H, W])
    # 定义特征图管理类
    map_master = mf.Map_master(map_sizes)
    # 定义列表用来存放样本框
    bottle = []
    cup = []
    bowl = []
    plate = []
    # 定义一个列表存放全部物体框
    GTmaster = [bottle, cup, bowl, plate]
    # 遍历全部特征图
    for feature_num in range(len(confs)):
        conf = confs[feature_num].detach().cpu()
        loc = locs[feature_num].detach().cpu()
        center = centers[feature_num].detach().cpu()
        # 对置信度分支进行抑制
        conf = conf * center
        # 根据置信度获取非背景区域
        indexes = torch.max(conf, 1)[1]
        indexes = indexes.numpy().tolist()[0]
        # 遍历置信度特征图，搜索超过阈值的点
        for i in range(len(indexes)):
            for j in range(len(indexes[i])):
                # 该点最大置信度超过阈值则为正样本
                if conf[0, indexes[i][j], i, j] >= cls_lime:
                    box = [feature_num, i, j, indexes[i][j], conf[0, indexes[i][j], i, j], loc[0, 0, i, j],
                           loc[0, 1, i, j], loc[0, 2, i, j], loc[0, 3, i, j]]
                    box = map_master.decode_coordinate(box, row, col)
                    GTmaster[indexes[i][j]].append(box)
    # 定义返回的物体盒子列表
    boxes = []
    # 依次遍历不同类别的列表并剔除无用的框
    for GT in GTmaster:
        if len(GT) == 0:
            continue
        while len(GT) > 0:
            max_obj = []
            for obj in GT[:]:
                # 寻找同一类别内置信度最高的框
                if max_obj == []:
                    max_obj = obj
                    continue
                if max_obj[1] < obj[1]:
                    max_obj = obj
            GT.remove(max_obj)
            # 将置信度最高的框选为物体盒子之一
            boxes.append(max_obj)
            if len(GT) > 0:
                # 删除与被选中框重合度较高的框
                for obj in GT[:]:
                    # 计算当前框与被选中物体盒子的交并比
                    iou = mf.compute_iou([obj[2], obj[3], obj[4], obj[5]],
                                         [max_obj[2], max_obj[3], max_obj[4], max_obj[5]])
                    if iou > iou_lime:
                        # 交并比超过阈值时，删除该框
                        GT.remove(obj)
    return boxes


if __name__ == '__main__':
    # 加载神经网络
    net = torch.load('.')
    net.eval()
    net.cuda()
    # load test set
    test_set = TestSet()
    loader = Data.DataLoader(
        dataset=test_set,  # torch TensorDataset format
        batch_size=1,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )
    # 逐图检测
    for step, label_paths in enumerate(loader):
        # 读取一帧
        frame = cv2.imread(label_paths[0].split(".")[0] + ".jpg")
        frame = cv2.resize(frame, (300, 300))
        row = frame.shape[0]
        col = frame.shape[1]
        torch_images, labels = get_image.get_label(label_paths)
        torch_images = torch_images.cuda()
        # 预测
        confs, locs, centers = net(torch_images)
        boxes = prediction(confs, locs, centers, 300, 300)
        for box in boxes:
            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]
            if box[0] == 0:
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 250, 250), 3)
            elif box[0] == 1:
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            elif box[0] == 2:
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            else:
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), 3)
        cv2.imshow('object detector', frame)
        if cv2.waitKey(0) & 0xFF == 27:
            break
