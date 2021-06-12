from src.dataset import create_dataloader, LoadImages
from src.config import JSON, IMAGE_SIZE
import albumentations as A
import src.my_albumentations as M
import matplotlib.pyplot as plt
import torch
from src.model import Model
import sys
from pathlib import Path

if __name__ == "__main__":
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

    _, dataloader = create_dataloader(
        "../_data/wallpapers", "test.json", transform=my_transform)
    imgs, labels = next(iter(dataloader))
    assert imgs.shape == (
        2, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]), f"dataset error {imgs.shape}"
    assert labels.shape == (2,), f"dataset error {labels.shape}"

    dataset = LoadImages(JSON, transform=img_transform)
    og_img, img, path = next(iter(dataset))
    assert img.shape == (
        3, IMAGE_SIZE[0], IMAGE_SIZE[1]), f"dataset error {img.shape}"

    print("dataset ok")

    img = torch.rand((4, 3, 1080, 1920))
    model = Model()
    pred = model(img)
    assert pred.shape == (4, 2), f"Model {pred.shape}"

    print("model ok")
