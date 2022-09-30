import torch
import torch.nn as nn
import map_function as mf


class IOUloss(nn.Module):
    def __init__(self):
        super(IOUloss, self).__init__()

    def forward(self, rect1, rect2):
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
        s_rect1 = (bottom1 - top1 + 1) * (right1 - left1 + 1)
        s_rect2 = (bottom2 - top2 + 1) * (right2 - left2 + 1)
        # 计算交叉矩形的各边坐标
        cross_left = max(left1, left2)
        cross_right = min(right1, right2)
        cross_top = max(top1, top2)
        cross_bottom = min(bottom1, bottom2)
        # 判断交叉矩形是否存在
        if cross_left >= cross_right or cross_top >= cross_bottom:
            # 交叉矩形不存在时
            return torch.tensor(0).type('torch.FloatTensor')
        else:
            # 交叉矩形存在时
            # 计算交叉矩形的面积
            s_cross = (cross_right - cross_left + 1) * (cross_bottom - cross_top + 1)
            if s_rect1 + s_rect2 - s_cross <= 0 or s_cross <= 0:
                return torch.tensor(0).type('torch.FloatTensor')
            # 计算并返回交并比
            return - torch.log(s_cross / (s_rect1 + s_rect2 - s_cross))


class FCOSloss(nn.Module):
    def __init__(self):
        super(FCOSloss, self).__init__()
        self.alpha = 0.25
        self.gamma = 2

    def forward(self, confses, locses, centerses, labels, device):
        # location,confidence:前向传播结果[loc,conf]
        # labels:标签
        # 获取batch_size
        batch_size = len(confses)
        # 定义IOU损失函数
        iou_loss = IOUloss()
        # 定义BCE损失函数
        center_loss = nn.BCELoss()
        # 定义批次损失
        loss = torch.tensor(0).type('torch.FloatTensor').to(device)
        ########## 对批次内的图像一个个进行处理并计算损失函数##########
        for img_num in range(batch_size):
            # 定义置信度损失
            loss_conf = torch.tensor(0).type('torch.FloatTensor').to(device)
            # 定义位置损失
            loss_l = torch.tensor(0).type('torch.FloatTensor').to(device)
            # 定义center抑制损失
            loss_center = torch.tensor(0).type('torch.FloatTensor').to(device)
            # 获取当前图片下的标签
            tags = labels[img_num]
            # 获取当前图片下的特征图列表
            confs = [confses[i][img_num] for i in range(5)]
            locs = [locses[i][img_num] for i in range(5)]
            centers = [centerses[i][img_num] for i in range(5)]
            # 获取全部特征图的尺寸
            map_sizes = []
            for map_num in range(len(confs)):
                # 获取特征图尺寸
                H = confs[map_num].size(1)
                W = confs[map_num].size(2)
                map_sizes.append([H, W])
            # 定义特征图管理类
            map_master = mf.Map_master(map_sizes)
            # 正样本数量
            poses = 0
            for feature_map in map_master.feature_maps:
                # 获取当前特征图
                conf = confs[feature_map.num]
                loc = locs[feature_map.num]
                center = centers[feature_map.num]
                # 计算置信度损失
                conf = torch.clamp(conf, min=0.00000001, max=0.99999999)
                loss_c = - (1 - self.alpha) * conf ** self.gamma * torch.log(1 - conf)
                for pixel in feature_map.pixels:
                    pixel.judge_pos(tags)
                    # 计算损失函数
                    # 计算置信度损失
                    for c_channel in range(conf.shape[0]):
                        if c_channel == pixel.tag[0]:
                            loss_c[c_channel, pixel.i, pixel.j] = (- self.alpha * (1 - conf[c_channel, pixel.i, pixel.j]) ** self.gamma) * torch.log(conf[c_channel, pixel.i, pixel.j])
                    if pixel.status == 1:
                        poses += 1
                        # 计算位置损失
                        loss_l = loss_l + iou_loss([pixel.tag[1], pixel.tag[2], pixel.tag[3], pixel.tag[4]],
                                                   [pixel.x - loc[0, pixel.i, pixel.j],
                                                    pixel.y - loc[1, pixel.i, pixel.j],
                                                    pixel.x + loc[2, pixel.i, pixel.j],
                                                    pixel.y + loc[3, pixel.i, pixel.j]])
                        # 计算抑制损失
                        loss_center = loss_center + center_loss(center[:, pixel.i, pixel.j],
                                                                torch.Tensor([pixel.center]).to(device))
                loss_conf += loss_c.sum()
            loss += loss_center + (loss_conf + loss_l) / poses if poses != 0 else loss_center + loss_conf + loss_l
#            if loss == - torch.log(torch.tensor(0).type('torch.FloatTensor')):
#                pass
        ###############################################################
        loss = loss / batch_size
        return loss
