import os
import numpy as np
import cv2
import map_function as mf
import xml.dom.minidom
import torch
import copy


def prediction(confs, locs, centers, row, col):
    """
    预测输入图片中工人安全帽佩戴情况
    :param confs: 置信度分之
    :param locs: 位置分之
    :param centers: 抑制分之
    :param row: 原图的行数
    :param col: 原图的列数
    :return:
    """
    iou_lime = 0.4  # 交并比阈值
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
    yellow = []
    red = []
    blue = []
    white = []
    none = []
    # 定义一个列表存放全部物体框
    GTmaster = [yellow, red, blue, white, none]
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
            for obj in GT:
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
                iou = 0  # 交并比
                temp = copy.deepcopy(GT)
                for obj in temp:
                    # 计算当前框与被选中物体盒子的交并比
                    iou = mf.compute_iou([obj[2], obj[3], obj[4], obj[5]],
                                         [max_obj[2], max_obj[3], max_obj[4], max_obj[5]])
                    if iou > iou_lime:
                        # 交并比超过阈值时，删除该框
                        GT.remove(obj)
    # 去除重合度过高的不同类别的冗余框
    final_boxes = []
    while len(boxes) > 0:
        max_obj = []
        for obj in boxes:
            # 寻找同一类别内置信度最高的框
            if max_obj == []:
                max_obj = obj
                continue
            if max_obj[1] < obj[1]:
                max_obj = obj
        boxes.remove(max_obj)
        # 将置信度最高的框选为物体盒子之一
        final_boxes.append(max_obj)
        if len(boxes) > 0:
            # 删除与被选中框重合度较高的框
            iou = 0  # 交并比
            temp = copy.deepcopy(boxes)
            for obj in temp:
                # 计算当前框与被选中物体盒子的交并比
                iou = mf.compute_iou([obj[2], obj[3], obj[4], obj[5]],
                                     [max_obj[2], max_obj[3], max_obj[4], max_obj[5]])
                if iou > 0.4 or iou == min([obj[4] * obj[5], max_obj[4] * max_obj[5]]) / max(
                        [obj[4] * obj[5], max_obj[4] * max_obj[5]]):
                    # 交并比超过阈值时，删除该框
                    boxes.remove(obj)
    return final_boxes


if __name__ == '__main__':
    # 初始化归一化层
    bn = torch.nn.InstanceNorm2d(3)
    label_list = ['yellow',
                  'red',
                  'blue',
                  'white',
                  'none']
    # 初始化特征提取器
    net = torch.load('./models/net18.pkl', map_location='cuda:0')
    device = torch.device("cuda:0")
    net.to(device)
    # label_path = './DataSet/test/'  # 测试集
    label_path = './DataSet/strange_label/'  # 企业测试集路径
    labels = os.listdir(label_path)
    # random.shuffle(labels)
    # 逐图检测
    for label in labels:
        # 读入xml文件
        dom = xml.dom.minidom.parse(label_path + label)
        # 得到文档元素对象
        root = dom.documentElement
        path = root.getElementsByTagName('path')[0]
        pathname = path.childNodes[0].data
        frame = cv2.imread(pathname)
        row = frame.shape[0]
        col = frame.shape[1]
        height = row
        width = col
        threshold = 960
        if height > threshold or width > threshold:
            if height > width:
                width = int((threshold / height) * width)
                height = threshold
            else:
                height = int((threshold / width) * height)
                width = threshold
        col = width
        row = height
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        image = cv2.GaussianBlur(frame, (5, 5), 0)
        torch_image = np.transpose(image, (2, 0, 1))
        torch_image = torch.from_numpy(torch_image)
        torch_image = torch_image.unsqueeze(0).type(torch.FloatTensor)
        torch_image = bn(torch_image).to(device)  # 归一化
        with torch.no_grad():
            try:
                confs, locs, centers = net([torch_image])
            except Exception as e:
                print(e)
                continue
            conf = confs[0]
            loc = locs[0]
            center = centers[0]
            try:
                boxes = prediction(conf, loc, center, height, width)
            except Exception as e:
                print(e)
                continue
            # 将预测结果以方框的形式贴在图片上
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
                elif box[0] == 3:
                    frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 3)
                else:
                    frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), 3)
            cv2.imshow('object detector', frame)
            if cv2.waitKey(0) & 0xFF == 27:
                break
