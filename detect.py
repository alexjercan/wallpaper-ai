# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import torch
import argparse
import albumentations as A
import src.my_albumentations as M

from src.config import parse_detect_config, DEVICE, read_yaml_config
from src.model import Model
from src.util import plot_predictions
from src.general import load_checkpoint
from src.dataset import LoadImages


def generatePredictions(model, dataset):
    for og_img, img, path in dataset:
        with torch.no_grad():
            img = img.to(DEVICE, non_blocking=True).unsqueeze(0)

            predictions = model(img)
            yield og_img, predictions, path


def detect(model=None, config=None):
    torch.backends.cudnn.benchmark = True

    config = parse_detect_config() if not config else config

    transform = A.Compose(
        [
            A.Resize(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
            A.Normalize(),
            M.MyToTensorV2(),
        ],
    )

    dataset = LoadImages(config.JSON, transform=transform)

    if not model:
        model = Model()
        model = model.to(DEVICE)
        _, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)

    model.eval()
    for img, predictions, path in generatePredictions(model, dataset):
        plot_predictions([img], predictions, [path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run inference on model')
    parser.add_argument('--detect', type=str, default="detect.yaml", help='detect config file')
    opt = parser.parse_args()

    config_detect = parse_detect_config(read_yaml_config(opt.detect))

    detect(config=config_detect)