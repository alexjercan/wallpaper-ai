# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import json
from copy import copy
import cv2

from torch.utils.data import Dataset, DataLoader
from util import load_image


def create_dataloader(dataset_root, json_path, batch_size=2, transform=None, workers=8, pin_memory=True, shuffle=True):
    dataset = Dataset(dataset_root, json_path, transform=transform)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, pin_memory=pin_memory, shuffle=shuffle)
    return dataset, dataloader


class Dataset(Dataset):
    def __init__(self, dataset_root, json_path, transform=None):
        super(Dataset, self).__init__()
        self.dataset_root = dataset_root
        self.json_path = os.path.join(dataset_root, json_path)
        self.transform = transform

        with open(self.json_path, "r") as f:
            self.json_data = json.load(f)

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        data = self.__load__(index)
        data = self.__transform__(data)
        return data

    def __load__(self, index):
        img_path = os.path.join(self.dataset_root, self.json_data[index]["image"])
        label = self.json_data[index]["label"]

        img = load_image(img_path)

        return img, label

    def __transform__(self, data):
        img, label = data

        if self.transform is not None:
            augmentations = self.transform(image=img)
            img = augmentations["image"]

        return img, label


class LoadImages():
    def __init__(self, json_data, transform=None):
        self.json_data = json_data
        self.transform = transform
        self.count = 0

    def __len__(self):
        return len(self.json_data)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        index = self.count

        if self.count == self.__len__():
            raise StopIteration
        self.count += 1

        data = self.__load__(index)
        data =  self.__transform__(data)
        return data

    def __load__(self, index):
        img_path = self.json_data[index]["image"]

        img = load_image(img_path)

        return img, img_path

    def __transform__(self, data):
        img, img_path = data
        og_img = copy(img)

        if self.transform is not None:
            augmentations = self.transform(image=img)
            img = augmentations["image"]

        return og_img, img, img_path


if __name__ == "__main__":
    from config import JSON, IMAGE_SIZE
    import albumentations as A
    import my_albumentations as M
    import matplotlib.pyplot as plt

    def visualize(image):
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(image)
        plt.show()

    my_transform = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
            A.Normalize(),
            M.MyToTensorV2(),
        ],
    )

    img_transform = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
            A.Normalize(),
            M.MyToTensorV2(),
        ],
    )

    _, dataloader = create_dataloader("../_data/wallpapers", "test.json", transform=my_transform)
    imgs, labels = next(iter(dataloader))
    assert imgs.shape == (2, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]), f"dataset error {imgs.shape}"
    assert labels.shape == (2,), f"dataset error {labels.shape}"

    dataset = LoadImages(JSON, transform=img_transform)
    og_img, img, path = next(iter(dataset))
    assert img.shape == (3, IMAGE_SIZE[0], IMAGE_SIZE[1]), f"dataset error {img.shape}"

    print("dataset ok")