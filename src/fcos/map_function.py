import math


def compute_iou(rect1, rect2):
    """
    功能：计算两个矩形的交并比损失
    参数：
        rect1:[left1, top1, right1, bottom1]
        rect2:[left2, top2, right2, bottom2]
    返回值：
        iou:计算出的交并比
    """
    # 获取矩形的四条边的位置
    left1 = rect1[0]
    top1 = rect1[1]
    right1 = rect1[2]
    bottom1 = rect1[3]
    left2 = rect2[0]
    top2 = rect2[1]
    right2 = rect2[2]
    bottom2 = rect2[3]
    # 分别计算两个矩形的面积
    s_rect1 = (bottom1 - top1) * (right1 - left1)
    s_rect2 = (bottom2 - top2) * (right2 - left2)
    # 计算交叉矩形的各边坐标
    cross_left = max(left1, left2)
    cross_right = min(right1, right2)
    cross_top = max(top1, top2)
    cross_bottom = min(bottom1, bottom2)
    # 判断交叉矩形是否存在
    if cross_left >= cross_right or cross_top >= cross_bottom:
        # 交叉矩形不存在时
        return 0
    else:
        # 交叉矩形存在时
        # 计算交叉矩形的面积
        s_cross = (cross_right - cross_left) * (cross_bottom - cross_top)
        # 计算并返回交并比
        return s_cross / (s_rect1 + s_rect2 - s_cross)


class Pixel:
    """
    一个特征点的类
    """

    def __init__(self, num, stride, i, j, thresholds):
        self.num = num  # 该点所属的特征图编号
        self.stride = stride
        self.status = 0  # 该点是否是正样本的状态码
        self.area = 0  # 匹配的标签的面积
        self.tag = [-1]  # 与该框匹配的标签框
        self.i = i  # 在其特征图上对应的行号（从0开始）
        self.j = j  # 在其特征图上对应的列号（从0开始）
        self.thresholds = thresholds  # 多尺度训练时的阈值
        self.center = 0  # 当该点为正样本时的抑制值
        # 对应到原图上的坐标
        self.x = int(stride / 2) + j * stride
        self.y = int(stride / 2) + i * stride

    def judge_pos(self, tags):
        """
        判断该像素点是不是一个正样本点
        :param tags: 全部标签
        :return:
        """
        for tag in tags:
            xmin = tag[1]
            ymin = tag[2]
            xmax = tag[3]
            ymax = tag[4]
            if xmin < self.x < xmax and ymin < self.y < ymax:
                # 计算该点距离各边的距离
                l = self.x - xmin
                t = self.y - ymin
                r = xmax - self.x
                b = ymax - self.y
                if self.thresholds[0] <= max(l, t, r, b) <= self.thresholds[1]:
                    if self.status == 0:
                        # 状态码变为1
                        self.status = 1
                        # 设置该点所在标签内的面积
                        self.area = (xmax - xmin + 1) * (ymax - ymin + 1)
                        # 设置该点的标签
                        self.tag = tag
                        # 设置该点的抑制值
                        self.center = math.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))
                    else:
                        # 若该点在两个标签框的相交区域，则讲该点匹配给面积较小的标签框
                        area = (xmax - xmin + 1) * (ymax - ymin + 1)
                        if self.area > area:
                            # 设置该点所在标签内的面积
                            self.area = (xmax - xmin + 1) * (ymax - ymin + 1)
                            # 设置该点的标签
                            self.tag = tag
                            # 设置该点的抑制值
                            self.center = math.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))


class Feature_map:
    """
    一个特征图的类
    """

    def __init__(self, num, row, col, stride, thresholds):
        self.num = num  # 特征图编号
        self.pixels = []  # 存储该特征图上特横点的列表
        self.row = row  # 该特征图的行数
        self.col = col  # 该特征图的列数
        self.stride = stride  # 该特征图的下采样倍数
        self.thresholds = thresholds  # 该特征图的多尺度阈值
        for i in range(self.row):
            for j in range(self.col):
                pixel = Pixel(self.num, self.stride, i, j, self.thresholds)
                self.pixels.append(pixel)


class Map_master:
    """
    一个生成和管理全部特征图上的特征点的类
    """

    def __init__(self, sizes):
        self.feature_maps = []  # 一个存储全部特征图的列表
        self.strides = [8, 16, 32, 64, 128]  # 各特征图的下采样倍数
        self.sizes = sizes  # 各特征图的尺寸
        self.num = 5  # 特征图数量
        self.thresholds = [0, 32, 64, 128, 256, float('inf')]
        for i in range(self.num):
            feature_map = Feature_map(i, self.sizes[i][0], self.sizes[i][1], self.strides[i],
                                      [self.thresholds[i], self.thresholds[i + 1]])
            self.feature_maps.append(feature_map)

    def decode_coordinate(self, tag, row, col):
        """
        将特征点转化为原图上的坐标
        :param tag: [num,i,j,c,mark,l,t,r,b]
        :param row: 原图的行数
        :param col: 原图的列数
        :return:
        """
        num = tag[0]  # 该特征点所处特征图编号
        i = tag[1]  # 该点所在特征图的行
        j = tag[2]  # 该点所在特征图的列
        c = tag[3]  # 该点的类别
        mark = tag[4]  # 该类别的置信度
        l = tag[5]
        t = tag[6]
        r = tag[7]
        b = tag[8]
        # 确定该点在原图上的位置
        x = int(self.strides[num] / 2) + j * self.strides[num]
        y = int(self.strides[num] / 2) + i * self.strides[num]
        # 确定该点的左上角和右下角
        xmin = max(int(x - l), 0)
        ymin = max(int(y - t), 0)
        xmax = min(int(x + r), col - 1)
        ymax = min(int(y + b), row - 1)
        return [c, mark, xmin, ymin, xmax, ymax]
