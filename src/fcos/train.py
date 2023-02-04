import torch
from DataLoader import FolderData
import get_image
from LossFunction import FCOSloss
from net import FCOS
import torch.utils.data as Data
from mAP import returnMAP

# torch.manual_seed(1)	#reproducible
if __name__ == '__main__':
    # hyper parameters
    BATCH_SIZE = 6
    EPOCH = 1000
    LR = 0.0001  # learning rate
    FT_LR = 0.000001  # learning rate for frozen parameters
    start = 0
    w_d = 0.005  # weigh decay
    optimizername = 'Adam'  # initialize optimizer

    # initialize model
    net = FCOS()
    # net = torch.load('./module/net'+ str(start-1) + '.pkl') # 43,49

    # initailize gpu for training and testing
    train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(train_device)
    test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # apply hyper-parameters
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

    # initialize optimizer and loss function
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

    # initialize training set
    torch_dataset = FolderData("./DataSet/labels/train/")
    # initialized dataloader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # shuffle the dataset
        num_workers=2,  # reading dataset by multi threads
    )
    # initialize maximum mAP
    max_mAP = 0
    # training
    for epoch in range(start, EPOCH):
        # release a mini-batch data
        for step, image_paths in enumerate(loader):
            # read images and labels
            torch_images, labels = get_image.get_label(image_paths)
            torch_images = torch_images.to(train_device)
            # obtain feature maps output by the model
            confs, locs, centers = net(torch_images)  # .to(train_device)
            # training
            loss = loss_func(confs, locs, centers, labels, train_device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch', epoch, 'Step:', step, '|train loss:%.4f' % loss)
        # save model
        # torch.save(classifier, './module/classifier' + str(epoch) + '.pkl')
        # evaluate the performance of current progress
        mAP = returnMAP(net)
        print('Epoch:', epoch, '|mAP:%.4f' % mAP)
        if mAP >= max_mAP:
            net.cpu()
            torch.save(net, './module/net' + str(epoch) + '.pkl')
            net.to(train_device)
            map_mAP = mAP
