import os
from torch.utils.data import Dataset

class FolderData(Dataset):

    def __init__(self, path):
        super().__init__()
        
        self.dataset_path = path
        # initialize label paths based on distribution
        self.label_paths = [self.dataset_path + xml for xml in os.listdir(self.dataset_path)] # Revise, use path.joinPath to avoid issues/gargage. And also validation tbh
        
    def __getitem__(self, index):
        # Return label of an image corresponding to its index.
        return self.label_paths[index]

    def __len__(self):
        return len(self.label_paths)
