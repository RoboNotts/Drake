from importlib.resources import read_text, open_binary
import torch
import xml.dom.minidom
import cv2
import fcos.map_function as mf
from fcos.DataLoader import FolderData
import torch.utils.data as Data
import fcos.get_image as get_image
import fcos.module
import fcos.net
from fcos.net import FCOS

def prediction(confs, locs, centers, row, col):
    # Find Classes.
    try:
        f = read_text(__package__, 'classes.txt')
        classes = f.splitlines()
    except FileNotFoundError:
        print("classes.txt file was not found...")
        exit(0)   
        
    iou_lime = 0.5  # threshold for iou
    cls_lime = 0.2  # threshold for confidence
    
    # obtain the size of all the feature maps
    map_sizes = []
    for map_num in range(len(confs)):
        # obtain the size of the feature map
        H = confs[map_num].size(2)
        W = confs[map_num].size(3)
        map_sizes.append([H, W])
    # initialize a manager for feature maps
    map_master = mf.Map_master(map_sizes)
    
    # initialize a list for storing predicted bounding boxes of different classes
    GTmaster = []
    for i in classes:
        GTmaster.append([])
    
    # traverse all feature maps
    for feature_num in range(len(confs)):
        conf = confs[feature_num].detach().cpu()
        loc = locs[feature_num].detach().cpu()
        center = centers[feature_num].detach().cpu()
        # suppress confidence
        conf = conf * center
        # obtain non-background area
        indexes = torch.max(conf, 1)[1]
        indexes = indexes.numpy().tolist()[0]
        # search for pixels on the feature map whose confidence are over threshold
        for i in range(len(indexes)):
            for j in range(len(indexes[i])):
                # the pixel is considered as positive sample if its confidence is larger than the threshold
                if conf[0, indexes[i][j], i, j] >= cls_lime:
                    box = [feature_num, i, j, indexes[i][j], conf[0, indexes[i][j], i, j], loc[0, 0, i, j],
                           loc[0, 1, i, j], loc[0, 2, i, j], loc[0, 3, i, j]]
                    box = map_master.decode_coordinate(box, row, col)
                    GTmaster[indexes[i][j]].append(box)
    # initialize a empty list for returning the final detected bounding boxes after NMS
    boxes = []
    # non maximum suppression (NMS)
    for GT in GTmaster:
        while len(GT) > 0:
            max_obj = []
            for obj in GT[:]:
                # obtain the bounding box with the highest confidence  within the same category
                if max_obj == []:
                    max_obj = obj
                    continue
                if max_obj[1] < obj[1]:
                    max_obj = obj
            GT.remove(max_obj)
            # select the bounding box of the highest confidence as a final predicted box
            boxes.append(max_obj)
            if len(GT) > 0:
                # remove other boxes of the same category whose iou between it and the selected box is larger than the threshold
                for obj in GT[:]:
                    # calculate the iou between it and the selected bounding box
                    iou = mf.compute_iou([obj[2], obj[3], obj[4], obj[5]],
                                         [max_obj[2], max_obj[3], max_obj[4], max_obj[5]])
                    if iou > iou_lime:
                        # delete it when the iou breaks the threshold
                        GT.remove(obj)
    return boxes

def main():
    # load class list
    print("balls")

    f = read_text('fcos', 'classes.txt')
    
    classes = f.splitlines()
    print("balls")
        
    # load the model
    with open_binary(fcos.module, 'net0.unpkl') as f:
        net = FCOS()
        net.load_state_dict(torch.load(f))
    net.eval()
    # load test set
    test_set = FolderData("./src/Drake/src/fcos/DataSet/labels/test/")
    loader = Data.DataLoader(
        dataset=test_set,  # torch TensorDataset format
        batch_size=1,  # mini batch size
        shuffle=True,  # shuffle the daatset
        num_workers=2,  # read data by multi threads
    )
    
    # detect
    for step, label_paths in enumerate(loader):
        # read one image
        xml_path = label_paths[0]
        # read annotation file
        dom = xml.dom.minidom.parse(xml_path)
        # obtain root of the xml file
        root = dom.documentElement
        objects = root.getElementsByTagName("object")
        path = root.getElementsByTagName('path')[0]
        # obtain the path of the image
        pathname = "./src/Drake/src/fcos/" + path.childNodes[0].data
        print(pathname)
        # read the image
        frame = cv2.imread(pathname)

        row = frame.shape[0]
        col = frame.shape[1]
        torch_images, labels = get_image.get_label(label_paths)
        # predict
        confs, locs, centers = net(torch_images)
        boxes = prediction(confs, locs, centers, row, col)
        for box in boxes:
            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]
            # draw rectangle
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 40, 255), 2)
            frame = cv2.putText(frame, classes[box[0]] + ":" + str(round(box[1].item(), 2)), (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 40, 255), 1)
            
        cv2.imshow('detections!', frame)
        if cv2.waitKey(0) & 0xFF == 27:
            break


if __name__ == '__main__':
    main()