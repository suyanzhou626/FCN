import torch
import random
import numpy as np


def occlusion(images, targets, length, occ_func, occ_p):
    return eval(occ_func)(images, targets, length, occ_p)


def fill_0(images, targets, length, occ_p):
    h = images.shape[2]
    w = images.shape[3]
    for i in range(images.shape[0]):
        if random.uniform(0, 1) > occ_p:
            continue

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        images[i, :, y1: y2, x1: x2] = 0.

    return images, targets


def fill_R(images, targets, length, occ_p):
    h = images.shape[2]
    w = images.shape[3]
    for i in range(images.shape[0]):
        if random.uniform(0, 1) > occ_p:
            continue

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        images[i, :, y1: y2, x1: x2] = torch.randn(y2 - y1, x2 - x1)

    return images, targets


def fill_next(images, targets, length, occ_p):
    h = images.shape[2]
    w = images.shape[3]
    for i in range(images.shape[0] - 1):
        if random.uniform(0, 1) > occ_p:
            continue

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        index = np.random.randint(0, images.shape[0] - 1)
        images[i, :, y1: y2, x1: x2] = images[index, :, y1: y2, x1: x2]

    return images, targets
