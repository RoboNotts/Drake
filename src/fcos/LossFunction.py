import torch
import torch.nn as nn
import map_function as mf


class IOUloss(nn.Module):
    def __init__(self):
        super(IOUloss, self).__init__()

    def forward(self, rect1, rect2):
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
        s_rect1 = (bottom1 - top1 + 1) * (right1 - left1 + 1)
        s_rect2 = (bottom2 - top2 + 1) * (right2 - left2 + 1)
        # calculate the coordinate of intersection rectangle
        cross_left = max(left1, left2)
        cross_right = min(right1, right2)
        cross_top = max(top1, top2)
        cross_bottom = min(bottom1, bottom2)
        # judge if the intersection exists
        if cross_left >= cross_right or cross_top >= cross_bottom:
            # the intersection does not exist
            return torch.tensor(0).type('torch.FloatTensor')
        else:
            # the intersection exists
            # calculate the area of the intersection
            s_cross = (cross_right - cross_left + 1) * (cross_bottom - cross_top + 1)
            if s_rect1 + s_rect2 - s_cross <= 0 or s_cross <= 0:
                return torch.tensor(0).type('torch.FloatTensor')
            # return iou loss
            return - torch.log(s_cross / (s_rect1 + s_rect2 - s_cross))


class FCOSloss(nn.Module):
    def __init__(self):
        super(FCOSloss, self).__init__()
        self.alpha = 0.25
        self.gamma = 2

    def forward(self, confses, locses, centerses, labels, device):
        # location,confidence: results of forward propaganda[loc,conf]
        # labels: annotations
        # obtain batch_size
        batch_size = len(labels)
        # initialize iou loss
        iou_loss = IOUloss()
        # initialize BCE loss
        center_loss = nn.BCELoss()
        # initialize overall loss of the minibatch
        loss = torch.tensor(0).type('torch.FloatTensor').to(device)
        ########## calculate loss function for each image ##########
        for img_num in range(batch_size):
            # initialize loss of confidence
            loss_conf = torch.tensor(0).type('torch.FloatTensor').to(device)
            # initialize loss of location
            loss_l = torch.tensor(0).type('torch.FloatTensor').to(device)
            # initialize loss of offset
            loss_center = torch.tensor(0).type('torch.FloatTensor').to(device)
            # the label of current image
            tags = labels[img_num]
            # obtain all the feature maps
            confs = [confses[i][img_num] for i in range(5)]
            locs = [locses[i][img_num] for i in range(5)]
            centers = [centerses[i][img_num] for i in range(5)]
            # obtain the sizes of all feature maps
            map_sizes = []
            for map_num in range(len(confs)):
                # obtain feature map size
                H = confs[map_num].size(1)
                W = confs[map_num].size(2)
                map_sizes.append([H, W])
            # initialize a manager of feature maps
            map_master = mf.Map_master(map_sizes)
            # the number of positive samples
            poses = 0
            for feature_map in map_master.feature_maps:
                # obtain current feature map
                conf = confs[feature_map.num]
                loc = locs[feature_map.num]
                center = centers[feature_map.num]
                # calculate the loss of confidence
                conf = torch.clamp(conf, min=0.00000001, max=0.99999999)
                loss_c = - (1 - self.alpha) * conf ** self.gamma * torch.log(1 - conf)
                for pixel in feature_map.pixels:
                    pixel.judge_pos(tags)
                    # calculate loss function
                    # calculate the loss of confidence
                    for c_channel in range(conf.shape[0]):
                        if c_channel == pixel.tag[0]:
                            loss_c[c_channel, pixel.i, pixel.j] = (- self.alpha * (1 - conf[c_channel, pixel.i, pixel.j]) ** self.gamma) * torch.log(conf[c_channel, pixel.i, pixel.j])
                    if pixel.status == 1:
                        poses += 1
                        # calculate loss of location
                        loss_l = loss_l + iou_loss([pixel.tag[1], pixel.tag[2], pixel.tag[3], pixel.tag[4]],
                                                   [pixel.x - loc[0, pixel.i, pixel.j],
                                                    pixel.y - loc[1, pixel.i, pixel.j],
                                                    pixel.x + loc[2, pixel.i, pixel.j],
                                                    pixel.y + loc[3, pixel.i, pixel.j]])
                        # calculate loss of offset
                        loss_center = loss_center + center_loss(center[:, pixel.i, pixel.j],
                                                                torch.Tensor([pixel.center]).to(device))
                loss_conf += loss_c.sum()
            loss += loss_center + (loss_conf + loss_l) / poses if poses != 0 else loss_center + loss_conf + loss_l
#            if loss == - torch.log(torch.tensor(0).type('torch.FloatTensor')):
#                pass
        ###############################################################
        loss = loss / batch_size
        return loss
