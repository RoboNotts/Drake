import os
import cv2
import numpy as np
import xml.dom.minidom
from DataAugmentation.TorchDataAugmentation import preprocessing
import torch


def get_label(label_ls, only_occluded=False):
    # 定义类别标签
    label_list = ['bottle', 'cup', 'bowl', 'plate']
    torch_images = []
    labels = []
    for label in label_ls:
        # 读入xml文件
        dom = xml.dom.minidom.parse(label)
        # 得到文档元素对象
        root = dom.documentElement
        objects = root.getElementsByTagName("object")
        path = root.getElementsByTagName('path')[0]
        pathname = "/home/wzl/final_project/" + path.childNodes[0].data
        image = cv2.imread(pathname)
        # obtain image size
        row = image.shape[0]
        col = image.shape[1]
        # image preprocessing
        torch_image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        torch_image = preprocessing(torch_image).unsqueeze(0)
        torch_images.append(torch_image)
        tags = []
        for obj in objects:
            # print ("*****Object*****")
            bndbox = obj.getElementsByTagName('bndbox')[0]
            name = obj.getElementsByTagName('name')[0]
            name_data = name.childNodes[0].data
            # print(name_data)
            xmin = bndbox.getElementsByTagName('xmin')[0]
            xmin_data = int(float(xmin.childNodes[0].data))
            # print(xmin_data)
            ymin = bndbox.getElementsByTagName('ymin')[0]
            ymin_data = int(float(ymin.childNodes[0].data))
            # print(ymin_data)
            xmax = bndbox.getElementsByTagName('xmax')[0]
            xmax_data = int(float(xmax.childNodes[0].data))
            # print(xmax_data)
            ymax = bndbox.getElementsByTagName('ymax')[0]
            ymax_data = int(float(ymax.childNodes[0].data))
            # print(ymax_data)
            if only_occluded:
                if "o_" in name_data:
                    continue
            tag = label_list.index(name_data[2:]) if "o_" in name_data else label_list.index(name_data)
            # print(tag)
            # 分别计算每个物体的中心点坐标和GT框长宽
            left = int(300 * xmin_data / col)
            top = int(300 * ymin_data / row)
            right = int(300 * xmax_data / col)
            bottom = int(300 * ymax_data / row)
            l = [tag, left, top, right, bottom]
            tags.append(l)
        labels.append(tags)
    torch_images = torch.cat(torch_images, dim=0)
    return torch_images, labels


def get_occluded_label(label_ls):
    # 定义类别标签
    label_list = ['bottle', 'cup', 'bowl', 'plate']
    torch_images = []
    labels = []
    for label in label_ls:
        # 读入xml文件
        dom = xml.dom.minidom.parse(label)
        # 得到文档元素对象
        root = dom.documentElement
        objects = root.getElementsByTagName("object")
        path = root.getElementsByTagName('path')[0]
        pathname = "/home/wzl/final_project/" + path.childNodes[0].data
        image = cv2.imread(pathname)
        # obtain image size
        row = image.shape[0]
        col = image.shape[1]
        # image preprocessing
        torch_image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        torch_image = preprocessing(torch_image).unsqueeze(0)
        torch_images.append(torch_image)
        tags = []
        for obj in objects:
            # print ("*****Object*****")
            bndbox = obj.getElementsByTagName('bndbox')[0]
            name = obj.getElementsByTagName('name')[0]
            name_data = name.childNodes[0].data
            # print(name_data)
            xmin = bndbox.getElementsByTagName('xmin')[0]
            xmin_data = int(float(xmin.childNodes[0].data))
            # print(xmin_data)
            ymin = bndbox.getElementsByTagName('ymin')[0]
            ymin_data = int(float(ymin.childNodes[0].data))
            # print(ymin_data)
            xmax = bndbox.getElementsByTagName('xmax')[0]
            xmax_data = int(float(xmax.childNodes[0].data))
            # print(xmax_data)
            ymax = bndbox.getElementsByTagName('ymax')[0]
            ymax_data = int(float(ymax.childNodes[0].data))
            # print(ymax_data)
            if "o_" in name_data:
                continue
            tag = label_list.index(name_data)
            # print(tag)
            # 分别计算每个物体的中心点坐标和GT框长宽
            left = int(300 * xmin_data / col)
            top = int(300 * ymin_data / row)
            right = int(300 * xmax_data / col)
            bottom = int(300 * ymax_data / row)
            l = [tag, left, top, right, bottom]
            tags.append(l)
        labels.append(tags)
    torch_images = torch.cat(torch_images, dim=0)
    return torch_images, labels


def addzeros(temp_labels, n):
    '''
    功能：对不整齐的numpy矩阵进行补零
    参数：
        temp_labels：输入不整齐的numpy二维矩阵列表,[array[],array[],...]
        n：单行0长度
    返回值：
        labels：输出整齐的numpy二维矩阵列表
    '''
    labels = []
    count = []  # 存储每个label内的物体数量
    for temp_label in temp_labels:
        # 获取当前label内的物体数量
        count.append(len(temp_label))
    # 对各图像物体数量不一做补零操作
    max_num = max(count)
    for tags in temp_labels:
        new_tags = []
        if len(tags) < max_num:
            # 当前标签内物体数小于最大值时，进行补零
            zero = [0, 0, 0, 0, 0]
            num = max_num - len(tags)
            for i in range(num):
                tags.append(zero)
        new_tags = tags
        labels.append(new_tags)
    return labels


def dezeros(l, n):
    '''
    功能：将补零的列表l消零
    参数：
        l：输入含零的列表
        n：单行0长度
    返回值：
        labels：输出不含零的列表
    '''
    labels = []
    zeros = [0 for _ in range(n)]
    # 消零
    for image in l:
        dezero_label = []
        for label in image:
            if label != zeros:
                dezero_label.append(label)
            else:
                break
        labels.append(dezero_label)
    return labels


if __name__ == '__main__':
    labels = os.listdir(label_path)
    i, l = get_label(labels)
    print(l)
    # print(i.sh)
    # l = torch.from_numpy(l)
    # print(l.shape)
    # print(l[0])
    cv2.imshow('show', np.transpose(i[1], (1, 2, 0)))
    cv2.waitKey(20000)
    # print(l[1])
    '''
    import anchor_function as af
    l = l.tolist()
    print(type(l[0]))
    #消零
    labels = dezeros(l,5)
    p,n = af.main_findindex(labels)
    p = af.decode_anchor(p[0],labels[0])
    print(p)
    print(labels[0])
    '''
