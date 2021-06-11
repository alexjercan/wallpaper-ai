# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, img_size=(1080, 1920)):
        super().__init__()
        h, w = img_size[0] // 4, img_size[1] // 4
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(16 * w * h, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.loss_val = 0

    def forward(self, predictions, labels):

        loss = self.loss(predictions, labels)

        self.loss_val = loss.item()

        return loss

    def show(self):
        loss = self.loss_val
        return f'(total:{loss:.4f})'

if __name__ == "__main__":
    img = torch.rand((4, 3, 1080, 1920))
    model = Model(img_size=(1080, 1920))
    pred = model(img)
    assert pred.shape == (4, 2), f"Model {pred.shape}"

    print("model ok")