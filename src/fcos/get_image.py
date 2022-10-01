import os
import cv2
import numpy as np
import xml.dom.minidom
from TorchDataAugmentation import preprocessing
import torch

# load class list
fh = open('./classes.txt', 'r')
# obtain class list
label_list = []
for line in fh:
    line = line.strip('\n')
    label_list.append(line)
fh.close()


def get_label(label_ls):
    # initialize a list to save input images as torch tensors
    torch_images = []
    # initialize a list to store the annotations of each image
    labels = []
    for label in label_ls:
        # read annotation file
        dom = xml.dom.minidom.parse(label)
        # obtain root of the xml file
        root = dom.documentElement
        objects = root.getElementsByTagName("object")
        path = root.getElementsByTagName('path')[0]
        pathname = "./" + path.childNodes[0].data
        image = cv2.imread(pathname)
        # image preprocessing
        torch_image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        torch_image = preprocessing(torch_image).unsqueeze(0)
        torch_images.append(torch_image)
        # analyse annotations
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
            tag = label_list.index(name_data)  # the num of the category, which starts with 0
            # print(tag)
            # obtain top left anf right bottom coordinate
            left = xmin_data
            top = ymin_data
            right = xmax_data
            bottom = ymax_data
            l = [tag, left, top, right, bottom]
            tags.append(l)
        labels.append(tags)
    torch_images = torch.cat(torch_images, dim=0)
    return torch_images, labels
