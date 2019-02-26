from __future__ import print_function, division
import os
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import numpy as np
# Ignore warnings
import warnings
import torch
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode


class AvenueDataset(Dataset):
    """Avenue dataset format."""

    def __init__(self, root_dir, transform=None):
        """
        Args:.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(os.path.join(root_dir, "labels.json")) as json_file:
            self.labels_file = json.load(json_file)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.labels_file[idx]["image"])
        image = io.imread(img_name)
        image = np.expand_dims(image, 2)

        if self.transform:
            image = self.transform(image)

        labels = self.labels_file[idx]
        return image, labels


class OnRoadObjectClassification(AvenueDataset):
    """Avenue dataset that return images with object on the road from three classes (boxes, balls and trashes)
    and the corresponding class of the object and distance."""

    def __getitem__(self, idx):
        image, labels = super(OnRoadObjectClassification, self).__getitem__(idx)
        return image, int(labels["object_class"][0] - 1), labels["object_distance"][0]


# Test dataset loading
if __name__ == '__main__':
    avenue_data = AvenueDataset("/tmp/ScenarioZoom_1548889499")
    #print("Data set length:" + str(len(avenue_data)))
    #print("First line:")
    print(avenue_data[0])
