# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import requests

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


def load_image_uri(path):
    response = requests.get(path)

    content = response.content
    nparr = np.fromstring(content, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def plot_predictions(images, predictions, paths):
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 200
    confs, predictions = torch.max(predictions, 1)
    confs = confs.cpu().numpy()
    predictions = predictions.cpu().numpy()

    for img, conf, pred, path in zip(images, confs, predictions, paths):
        fig = plt.figure()
        fig.suptitle(f'{path} {pred}:{conf}')
        plt.imshow(img)
        plt.show()
