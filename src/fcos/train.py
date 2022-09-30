import torch
from DataLoader import TrainSet
import get_image
from LossFunction import FCOSloss
from net import FCOS
import torch.utils.data as Data

# torch.manual_seed(1)	#reproducible

# 超参数
BATCH_SIZE = 16
EPOCH = 1000
LR = 0.0001  # 学习RATE
FT_LR = 0.000001  # Fine Tuning部分学习率
start = 11
w_d = 0.005
optimizername = 'Adam'

# 初始化神经网络
# net = FCOS()
net = torch.load('./models/net'+ str(start-1) + '.pkl') # 43,49

# 初始化训练和测试gpu
train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(train_device)
test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 设置权重衰减
weight_p, bias_p, FT_weight_p, FT_bias_p, feat_weight_p, feat_bias_p = [], [], [], [], [], []

for name, p in net.named_parameters():
    if 'FT' in name:
        if 'bias' in name:
            FT_bias_p += [p]
        else:
            FT_weight_p += [p]
    else:
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

# 初始化optimizer和损失函数
if optimizername == 'SGD':
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay': w_d, 'lr': LR},
                                 {'params': bias_p, 'weight_decay': 0, 'lr': LR},
                                 {'params': FT_weight_p, 'weight_decay': w_d, 'lr': FT_LR},
                                 {'params': FT_bias_p, 'weight_decay': 0, 'lr': FT_LR},
                                 ], momentum=0.9)
elif optimizername == 'Adam':
    optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay': w_d, 'lr': LR},
                                  {'params': bias_p, 'weight_decay': 0, 'lr': LR},
                                  {'params': FT_weight_p, 'weight_decay': w_d, 'lr': FT_LR},
                                  {'params': FT_bias_p, 'weight_decay': 0, 'lr': FT_LR},
                                  ])
loss_func = FCOSloss()

# 初始化数据集
torch_dataset = TrainSet()
# 把dataset放入DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=2,  # 多线程来读数据
)
# 训练所有整套数据EPOCH次
for epoch in range(start, EPOCH):
    # 每次释放BATCH_SIZE个图片路径
    for step, image_paths in enumerate(loader):
        # 读取全部图片和标签
        torch_images, labels = get_image.get_label(image_paths)
        torch_images = torch_images.to(train_device)
        # 获取该图片的特征图
        confs, locs, centers = net(torch_images)  # .to(train_device)
        # 训练
        loss = loss_func(confs, locs, centers, labels, train_device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch:', epoch, '|train loss:%.4f' % loss)
    # 保存训练好的神经
    # torch.save(classifier, './models/classifier' + str(epoch) + '.pkl')
    torch.save(net, './models/net' + str(epoch) + '.pkl')
