# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

from metrics import MetricFunction, print_single_error
import os
import re

import torch
import torch.optim
import argparse
import albumentations as A
import my_albumentations as M

from tqdm import tqdm
from config import IMAGE_SIZE, parse_test_config, parse_train_config, DEVICE, read_yaml_config
from datetime import datetime as dt
from model import Model, LossFunction
from test import test
from general import tensors_to_device, save_checkpoint, load_checkpoint
from dataset import create_dataloader


def train_one_epoch(model, dataloader, loss_fn, metric_fn, solver, epoch_idx):
    loop = tqdm(dataloader, position=0, leave=True)

    for i, tensors in enumerate(loop):
        imgs, labels = tensors_to_device(tensors, DEVICE)

        predictions = model(imgs)

        loss = loss_fn(predictions, labels)
        metric_fn.evaluate(predictions, labels)

        model.zero_grad()
        loss.backward()
        solver.step()

        loop.set_postfix(loss=loss_fn.show(), epoch=epoch_idx)
    loop.close()

def train(config=None, config_test=None):
    torch.backends.cudnn.benchmark = True

    config = parse_train_config() if not config else config

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

    model = Model(config.IMAGE_SIZE)
    solver = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=config.LEARNING_RATE, betas=config.BETAS,
                              eps=config.EPS, weight_decay=config.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(solver, milestones=config.MILESTONES, gamma=config.GAMMA)
    model = model.to(DEVICE)

    loss_fn = LossFunction()

    epoch_idx = 0
    if config.CHECKPOINT_FILE and config.LOAD_MODEL:
        epoch_idx, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)

    output_dir = os.path.join(config.OUT_PATH, re.sub("[^0-9a-zA-Z]+", "-", dt.now().isoformat()))

    for epoch_idx in range(epoch_idx, config.NUM_EPOCHS):
        metric_fn = MetricFunction(config.BATCH_SIZE)

        model.train()
        train_one_epoch(model, dataloader, loss_fn, metric_fn, solver, epoch_idx)
        print_single_error(epoch_idx, loss_fn.show(), metric_fn.show())
        lr_scheduler.step()

        if config.TEST:
            test(model, config_test)
        if config.SAVE_MODEL:
            save_checkpoint(epoch_idx, model, output_dir)

    if not config.TEST:
        test(model, config_test)
    if not config.SAVE_MODEL:
        save_checkpoint(epoch_idx, model, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--train', type=str, default="train.yaml", help='train config file')
    parser.add_argument('--test', type=str, default="test.yaml", help='test config file')
    opt = parser.parse_args()

    config_train = parse_train_config(read_yaml_config(opt.train))
    config_test = parse_test_config(read_yaml_config(opt.test))

    train(config_train, config_test)