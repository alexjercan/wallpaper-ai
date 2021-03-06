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

from tqdm import tqdm
from src.metrics import MetricFunction, print_single_error
from src.config import parse_test_config, DEVICE, read_yaml_config
from src.model import Model, LossFunction
from src.general import load_checkpoint, tensors_to_device
from src.dataset import create_dataloader


def run_test(model, dataloader, loss_fn, metric_fn):
    loop = tqdm(dataloader, position=0, leave=True)

    for _, tensors in enumerate(loop):
        imgs, labels = tensors_to_device(tensors, DEVICE)
        with torch.no_grad():
            predictions = model(imgs)

            loss_fn(predictions, labels)
            metric_fn.evaluate(predictions, labels)
    loop.close()


def test(model=None, config=None):
    epoch = 0
    torch.backends.cudnn.benchmark = True

    config = parse_test_config() if not config else config

    transform = A.Compose(
        [
            A.Resize(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
            A.Normalize(),
            M.MyToTensorV2(),
        ],
    )

    _, dataloader = create_dataloader(config.DATASET_ROOT, config.JSON_PATH,
                                      batch_size=config.BATCH_SIZE, transform=transform,
                                      workers=config.WORKERS, pin_memory=config.PIN_MEMORY, shuffle=config.SHUFFLE)

    if not model:
        model = Model()
        model = model.to(DEVICE)
        epoch, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)

    loss_fn = LossFunction()
    metric_fn = MetricFunction(config.BATCH_SIZE)

    model.eval()
    run_test(model, dataloader, loss_fn, metric_fn)
    print_single_error(epoch, loss_fn.show(), metric_fn.show())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('--test', type=str, default="test.yaml", help='test config file')
    opt = parser.parse_args()

    config_test = parse_test_config(read_yaml_config(opt.test))

    test(config=config_test)