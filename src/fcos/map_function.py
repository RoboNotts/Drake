import math


def compute_iou(rect1, rect2):
    """
    calculate iou between two bounding boxes
    :param：
        rect1:[left1, top1, right1, bottom1]
        rect2:[left2, top2, right2, bottom2]
    return：
        iou
    """
    # obtain the index of boundaries
    left1 = rect1[0]
    top1 = rect1[1]
    right1 = rect1[2]
    bottom1 = rect1[3]
    left2 = rect2[0]
    top2 = rect2[1]
    right2 = rect2[2]
    bottom2 = rect2[3]
    # calculate the area of teh two bounding boxes, respectively
    s_rect1 = (bottom1 - top1) * (right1 - left1)
    s_rect2 = (bottom2 - top2) * (right2 - left2)
    # calculate the coordinate of intersection rectangle
    cross_left = max(left1, left2)
    cross_right = min(right1, right2)
    cross_top = max(top1, top2)
    cross_bottom = min(bottom1, bottom2)
    # judge if the intersection exists
    if cross_left >= cross_right or cross_top >= cross_bottom:
        # the intersection does not exist
        return 0
    else:
        # the intersection exists
        # calculate the area of the intersection
        s_cross = (cross_right - cross_left) * (cross_bottom - cross_top)
        # return iou
        return s_cross / (s_rect1 + s_rect2 - s_cross)


class Pixel:
    """
    class for a pixel on the feature map
    """

    def __init__(self, num, stride, i, j, thresholds):
        self.num = num  # the num of the feature map where the pixel is located
        self.stride = stride
        self.status = 0  # marker to show if the pixel is a positive sample
        self.area = 0  # the area of the bounding box matching the pixel
        self.tag = [-1]  # the label of the boundnig box
        self.i = i  # the horizontal index of the pixel on the feature map（start from 0）
        self.j = j  # the vertical index of the pixel on the feature map（start from 0））
        self.thresholds = thresholds  # the threshold for multiscale training
        self.center = 0  # the offset when the pixel is a positive sample
        # the corresponding coordinate of the pixel under teh scale of the raw input image
        self.x = int(stride / 2) + j * stride
        self.y = int(stride / 2) + i * stride

    def judge_pos(self, tags):
        """
        judge if the pixel is a positiev sample
        :param tags: all labels
        :return:
        """
        for tag in tags:
            xmin = tag[1]
            ymin = tag[2]
            xmax = tag[3]
            ymax = tag[4]
            if xmin < self.x < xmax and ymin < self.y < ymax:
                # calculate the distance between the pixel and four boundries
                l = self.x - xmin
                t = self.y - ymin
                r = xmax - self.x
                b = ymax - self.y
                if self.thresholds[0] <= max(l, t, r, b) <= self.thresholds[1]:
                    if self.status == 0:
                        # change status to 1
                        self.status = 1
                        # set the area of the bounding box
                        self.area = (xmax - xmin + 1) * (ymax - ymin + 1)
                        # set the label of the pixel
                        self.tag = tag
                        # set the offset
                        self.center = math.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))
                    else:
                        # if this pixel is situated in the intersection of two bounding box,
                        # then assign it to the one with smaller area
                        area = (xmax - xmin + 1) * (ymax - ymin + 1)
                        if self.area > area:
                            # set the area of the bounding box
                            self.area = (xmax - xmin + 1) * (ymax - ymin + 1)
                            # set the label of the pixel
                            self.tag = tag
                            # set the offset
                            self.center = math.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))


class Feature_map:
    """
    class for a feature map
    """

    def __init__(self, num, row, col, stride, thresholds):
        self.num = num  # the num of the feature map
        self.pixels = []  # a list containing all the pixels of the feature map
        self.row = row  # the number of rows of the feature map
        self.col = col  # the number of columns of the feature map
        self.stride = stride  # the down sample rate of the feature map
        self.thresholds = thresholds  # the threshold for multiscale training
        for i in range(self.row):
            for j in range(self.col):
                pixel = Pixel(self.num, self.stride, i, j, self.thresholds)
                self.pixels.append(pixel)


class Map_master:
    """
    a class for generating and managing all faeture maps
    """

    def __init__(self, sizes):
        self.feature_maps = []  # a list contaiinng all feature maps
        self.strides = [8, 16, 32, 64, 128]  # the down sample rates of all feature maps
        self.sizes = sizes  # the size of each feature map
        self.num = 5  # the number of feature maps
        self.thresholds = [0, 32, 64, 128, 256, float('inf')]
        for i in range(self.num):
            feature_map = Feature_map(i, self.sizes[i][0], self.sizes[i][1], self.strides[i],
                                      [self.thresholds[i], self.thresholds[i + 1]])
            self.feature_maps.append(feature_map)

    def decode_coordinate(self, tag, row, col):
        """
        transfer the coordinate from feature map scale to input image scale
        :param tag: [num,i,j,c,mark,l,t,r,b]
        :param row: the number of rows of raw image
        :param col: the number of columns of raw image
        :return:
        """
        num = tag[0]  # the index of the feature map where the pixel is located at
        i = tag[1]  # the horizontal index of the pixel
        j = tag[2]  # the vertical index of the pixel
        c = tag[3]  # the class of the pixel
        mark = tag[4]  # the confidence of the class of the pixel
        l = tag[5]
        t = tag[6]
        r = tag[7]
        b = tag[8]
        # calculate the coordinate of the pixel under input raw image scale
        x = int(self.strides[num] / 2) + j * self.strides[num]
        y = int(self.strides[num] / 2) + i * self.strides[num]
        # calculate the top left and right bottom coordinate of the pixel under input image scale
        xmin = max(int(x - l), 0)
        ymin = max(int(y - t), 0)
        xmax = min(int(x + r), col - 1)
        ymax = min(int(y + b), row - 1)
        return [c, mark, xmin, ymin, xmax, ymax]
