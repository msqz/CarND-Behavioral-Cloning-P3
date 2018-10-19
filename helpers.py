#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import csv
import numpy as np
import pickle
from sklearn.utils import shuffle
from copy import deepcopy
import cv2
import keras.backend as K
from math import ceil

CORRECTION = 0.2


def visualize(model, from_layer, to_layer, data, labels):
    result = K.function([model.get_layer(from_layer).input],
                        [model.get_layer(to_layer).output])([data])[0]

    fig, ax = plt.subplots(ceil(len(result)/5), 5)
    for i, r in enumerate(result):
        row = ceil(i/5) - 1
        i == 0 and row == 0
        col = i % 5
        ax[row][col].imshow(r.astype(np.uint8))
        ax[row][col].set_title(labels[i])
    plt.show()


def read_lines(*paths):
    lines = []
    for path in paths:
        with open(path) as f:
            reader = csv.reader(f)
            for line in reader:
                center_measurement = round(float(line[3]), 2)
                lines.append((line[0].strip(), center_measurement))
                # Min/max - to kepp from exceeding angle (-1.0, 1.0)
                left_measurement = round(center_measurement + CORRECTION, 2)
                lines.append((line[1].strip(), min(1, left_measurement)))
                right_measurement = round(center_measurement - CORRECTION, 2)
                lines.append((line[2].strip(), max(-1, right_measurement)))

    expansion1 = deepcopy(lines)
    for line in expansion1:
        line += ('exp1',)

    expansion2 = deepcopy(lines)
    for line in expansion2:
        line += ('exp2',)

    expansion3 = deepcopy(lines)
    for line in expansion2:
        line += ('exp3',)

    return lines + expansion1 + expansion2 + expansion3


def load(lines, limit=None, offset=None):
    start = offset or 0
    end = (limit and start + limit) or len(lines)

    images = []
    measurements = []
    for line in lines[start:end]:
        images.append(plt.imread(line[0]))
        measurements.append(line[1])
        if line[-1] == 'exp1':
            images[-1] = np.fliplr(images[-1])
            measurements[-1] = -measurements[-1]
        if line[-1] == 'exp2':
            images[-1] = cv2.blur(images[-1], (3, 3)).astype(np.uint8)
        if line[-1] == 'exp3':
            images[-1] = cv2.blur(np.fliplr(images[-1]),
                                  (3, 3)).astype(np.uint8)
            measurements[-1] = -measurements[-1]

    return np.array(images), np.array(measurements)


def generator(lines, batch_size=32):
    while 1:
        for offset in range(0, len(lines), batch_size):
            X_train, y_train = load(lines, batch_size, offset)
            yield X_train, y_train
