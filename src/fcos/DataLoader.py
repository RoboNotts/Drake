import pickle
from torch.utils.data import Dataset
import random


class TrainSet(Dataset):
    def __init__(self):
        """
        training set
        """
        super(TrainSet, self).__init__()
        # open the label path file of training set
        fh = open('/home/wzl/final_project/train.pkl', 'rb')
        train = pickle.load(fh)
        fh.close()
        # initialize dataset path
        self.dataset_root = "/home/wzl/final_project/"
        # initialize distribution
        self.distribution = {
            "none": 1200,
            "slight": 1200,
            "medium": 1200,
            "heavy": 1200
        }
        # initialize label paths based on distribution
        self.label_paths = []
        for extent, num in self.distribution.items():
            self.label_paths += train[extent][:num]
        # shuffle dataset
        random.shuffle(self.label_paths)

    def __getitem__(self, index):
        """
        according to index obtain image and its label
        :param index:
        :return:
        """
        return self.dataset_root + self.label_paths[index]

    def __len__(self):
        return len(self.label_paths)


class TestSet(Dataset):
    def __init__(self):
        """
        training set
        """
        super(TestSet, self).__init__()
        # open the label path file of training set
        fh = open('/home/wzl/final_project/test.pkl', 'rb')
        test = pickle.load(fh)
        fh.close()
        # initialize dataset path
        self.dataset_root = "/home/wzl/final_project/"
        # initialize label paths based on distribution
        self.label_paths = []
        # self.returnAll(test)
        # self.returnByExtent(test, "heavy")
        self.returnByLocation(test, "bottom")
        # self.returnByType(test, "different")
        # self.returnByExtent(test, "none")
        # shuffle dataset
        random.shuffle(self.label_paths)

    def __getitem__(self, index):
        """
        according to index obtain image and its label
        :param index:
        :return:
        """
        return self.dataset_root + self.label_paths[index]

    def __len__(self):
        return len(self.label_paths)

    def returnByExtent(self, test, target_extent):
        """
        return paths of samples belonging to the target extent
        """
        for extent in test.keys():
            if target_extent != "none":
                if extent == target_extent:
                    for category in test[extent].keys():
                        for occlusion_location in test[extent][category].keys():
                            for occlusion_type in test[extent][category][occlusion_location].keys():
                                self.label_paths += test[extent][category][occlusion_location][occlusion_type]
            else:
                if extent == "none":
                    for category in test[extent].keys():
                        self.label_paths += test[extent][category]

    def returnAll(self, test):
        """
        return paths of all samples
        """
        for extent in test.keys():
            if extent != "none":
                for category in test[extent].keys():
                    for occlusion_location in test[extent][category].keys():
                        for occlusion_type in test[extent][category][occlusion_location].keys():
                            self.label_paths += test[extent][category][occlusion_location][occlusion_type]
            else:
                for category in test[extent].keys():
                    self.label_paths += test[extent][category]
    def returnByLocation(self, test, target_location):
        """
        return paths of samples belonging to the target extent
        """
        for extent in test.keys():
            if extent != "none":
                for category in test[extent].keys():
                    for occlusion_location in test[extent][category].keys():
                        if occlusion_location == target_location:
                            for occlusion_type in test[extent][category][occlusion_location].keys():
                                self.label_paths += test[extent][category][occlusion_location][occlusion_type]

    def returnByType(self, test, target_type):
        """
        return paths of samples belonging to the target extent
        """
        for extent in test.keys():
            if extent != "none":
                for category in test[extent].keys():
                    for occlusion_location in test[extent][category].keys():
                        for occlusion_type in test[extent][category][occlusion_location].keys():
                            if occlusion_type == target_type:
                                self.label_paths += test[extent][category][occlusion_location][occlusion_type]