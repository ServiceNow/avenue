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
        img_name = os.path.join(os.path.join(self.root_dir, "rgb"),self.labels_file[idx]["image"])
        image = io.imread(img_name)
        image = np.expand_dims(image, 2)

        if self.transform:
            image = self.transform(image)

        labels = self.labels_file[idx]
        return image, labels


class AvenueDatasetSegmentation(Dataset):
    """Avenue dataset format."""

    dict_segmentation = {
        "road" : 7,
        "sidewalk": 8,
        "building" : 11,
        "sign" : 20,
        "car" : 26,
        "fence": 29,
        "pedestrian": 24,
        "dumb_pedestrian": 3,
        "player": 12,
        "light": 17,
        "decor": 4
    }

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

    def __getitem__(self, idx, mask_list=None):
        rgb_name = os.path.join(os.path.join(self.root_dir, "rgb"), self.labels_file[idx]["image"]) + ".jpg"
        rgb = io.imread(rgb_name)
        rgb = np.expand_dims(rgb, 2)
        segmentation_name = os.path.join(os.path.join(self.root_dir, "segmentation"), self.labels_file[idx]["image"]) + ".png"
        segmentation = io.imread(segmentation_name)
        segmentation = np.expand_dims(segmentation, 2)

        # Filter segmentation
        if mask_list is not None:
            mask = np.zeros_like(segmentation)
            for i in range(len(mask_list)):
                mask = mask and (segmentation[:, :, 0] == self.dict_segmentation[mask_list[i]])

        if self.transform:
            rgb = self.transform(rgb)
            segmentation = self.transform(segmentation)

        labels = self.labels_file[idx]
        return rgb, segmentation, labels


# Specialization of the dataloader
class OnRoadObjectClassification(AvenueDataset):
    """Avenue dataset that return images with object on the road from three classes (boxes, balls and trashes)
    and the corresponding class of the object and distance."""

    def __getitem__(self, idx):
        image, labels = super(OnRoadObjectClassification, self).__getitem__(idx)
        return image, int(labels["object_class"][0]), labels["object_distance"][0]


# Test dataset loading
if __name__ == '__main__':
    avenue_data = AvenueDataset("/tmp/Humanware_v1_1551306484")
    data_row = avenue_data[10]
