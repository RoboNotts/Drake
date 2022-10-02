import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
# from torchsummary import summary
import numpy as np
import math


class FCOS(nn.Module):
    def __init__(self):
        super(FCOS, self).__init__()
        model = models.resnet50(pretrained=False)
        pre = torch.load('./src/Drake/src/fcos/resnet50-19c8e357.pth')
        model.load_state_dict(pre)

        ########## stage1 ##########
        self.stage1_FT = nn.Sequential()

        ########## stage2 ##########
        self.stage2_FT = nn.Sequential()

        ########## stage3 ##########
        self.stage3_FT = nn.Sequential()

        ########## stage4 ##########
        self.stage4_FT = nn.Sequential()

        ########## stage5 ##########
        self.stage5_FT = nn.Sequential()

        for idx, m in enumerate(model.children()):
            if idx < 4:
                self.stage1_FT.add_module(str(idx), m)
            elif idx == 4:
                self.stage2_FT.add_module(str(idx), m)
            elif idx == 5:
                self.stage3_FT.add_module(str(idx), m)
            elif idx == 6:
                self.stage4_FT.add_module(str(idx), m)
            elif idx == 7:
                self.stage5_FT.add_module(str(idx), m)

        self.P5 = nn.Sequential(
            nn.Conv2d(
                in_channels=2048,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.C4 = nn.Sequential(
            nn.Conv2d(
                in_channels=1024,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.P4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.C3 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.P3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.P6 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.P7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.head7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conf7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=22,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )
        self.loc7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),  # 激活
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.center7 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid()
        )

        self.head6 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conf6 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=22,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )
        self.loc6 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.center6 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid()
        )

        self.head5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conf5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=22,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )

        self.loc5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.center5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid()
        )

        self.head4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conf4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=22,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )

        self.loc4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.center4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid()
        )

        self.head3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conf3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=22,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )

        self.loc3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.center3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid()
        )

    def forward(self, input_images):
        stage1 = self.stage1_FT(input_images)
        stage2 = self.stage2_FT(stage1)
        stage3 = self.stage3_FT(stage2)
        stage4 = self.stage4_FT(stage3)
        stage5 = self.stage5_FT(stage4)
        P5 = self.P5(stage5)
        P6 = self.P6(P5)
        P7 = self.P7(P6)
        P4 = self.P4(self.C4(stage4) + F.interpolate(P5, size=[stage4.shape[2], stage4.shape[3]]))
        P3 = self.P3(self.C3(stage3) + F.interpolate(P4, size=[stage3.shape[2], stage3.shape[3]]))

        head3 = self.head3(P3)
        conf3 = self.conf3(head3)
        loc3 = torch.exp(self.loc3(P3))
        center3 = self.center3(head3)

        head4 = self.head4(P4)
        conf4 = self.conf4(head4)
        loc4 = torch.exp(self.loc4(P4))
        center4 = self.center4(head4)

        head5 = self.head5(P5)
        conf5 = self.conf5(head5)
        loc5 = torch.exp(self.loc5(P5))
        center5 = self.center5(head5)

        head6 = self.head6(P6)
        conf6 = self.conf6(head6)
        loc6 = torch.exp(self.loc6(P6))
        center6 = self.center6(head6)

        head7 = self.head7(P7)
        conf7 = self.conf7(head7)
        loc7 = torch.exp(self.loc7(P7))
        center7 = self.center7(head7)

        confs = [conf3, conf4, conf5, conf6, conf7]
        locs = [loc3, loc4, loc5, loc6, loc7]
        centers = [center3, center4, center5, center6, center7]
        return confs, locs, centers




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# ssd = ssd.to(device)
# summary(ssd,input_size=(3,300,300))

# csp = FCOS()
# print(csp)  # net architecture
#summary(csp, input_size=(3, 300, 300), device='cpu')
