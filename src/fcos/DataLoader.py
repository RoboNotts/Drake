import os
import pickle
from torch.utils.data import Dataset
import random


class TrainSet(Dataset):
    def __init__(self):
        """
        training set
        """
        super(TrainSet, self).__init__()
        # initialize dataset path
        self.dataset_root = "./DataSet/labels/train/"
        # initialize label paths based on distribution
        self.label_paths = [self.dataset_root + xml_file for xml_file in os.listdir(self.dataset_root)]
        # shuffle dataset
        random.shuffle(self.label_paths)

    def __getitem__(self, index):
        """
        according to index obtain image and its label
        :param index:
        :return:
        """
        return self.label_paths[index]

    def __len__(self):
        return len(self.label_paths)


class TestSet(Dataset):
    def __init__(self):
        """
        test set
        """
        super(TestSet, self).__init__()
        # initialize dataset path
        self.dataset_root = "./DataSet/labels/test/"
        # initialize label paths based on distribution
        self.label_paths = [self.dataset_root + xml_file for xml_file in os.listdir(self.dataset_root)]
        # shuffle dataset
        random.shuffle(self.label_paths)

    def __getitem__(self, index):
        """
        according to index obtain image and its label
        :param index:
        :return:
        """
        return self.label_paths[index]

    def __len__(self):
        return len(self.label_paths)


class ValSet(Dataset):
    def __init__(self):
        """
        validation set
        """
        super(ValSet, self).__init__()
        # initialize dataset path
        self.dataset_root = "./DataSet/labels/val/"
        # initialize label paths based on distribution
        self.label_paths = [self.dataset_root + xml_file for xml_file in os.listdir(self.dataset_root)]
        # shuffle dataset
        random.shuffle(self.label_paths)

    def __getitem__(self, index):
        """
        according to index obtain image and its label
        :param index:
        :return:
        """
        return self.label_paths[index]

    def __len__(self):
        return len(self.label_paths)