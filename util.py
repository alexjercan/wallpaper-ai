# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import cv2
import torch
import matplotlib.pyplot as plt

def load_image(path):
    img = img2rgb(path)  # RGB
    assert img is not None, 'Image Not Found ' + path
    return img


def img2rgb(path):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def plot_predictions(images, predictions, paths):
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 200
    print(predictions)
    confs, predictions = torch.max(predictions, 1)
    confs = confs.cpu().numpy()
    predictions = predictions.cpu().numpy()

    for img, conf, pred, path in zip(images, confs, predictions, paths):
        fig = plt.figure()
        fig.suptitle(f'{path} {pred}:{conf}')
        plt.imshow(img)
        plt.show()
