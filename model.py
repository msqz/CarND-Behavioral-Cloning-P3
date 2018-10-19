#!/usr/bin/env python3
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, Dropout, MaxPool2D, Cropping2D, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import helpers
import sys


def balance(lines):
    balanced = []
    hist = np.histogram([line[1] for line in lines], 100)
    bins_len = len(hist[0])
    mean = np.mean(hist[0], 0)
    ranges = [(hist[1][i], hist[1][i+1]) for i in range(bins_len)]
    grouped = {}
    for i in range(bins_len):
        grouped[i] = []

    for line in lines:
        for i, r in enumerate(ranges):
            if r is ranges[-1] and r[0] <= line[1] <= r[1]:
                grouped[i].append(line)
            elif r[0] <= line[1] < r[1]:
                grouped[i].append(line)

    balanced = [grouped[i][:int(mean)] for i in range(bins_len)]

    return [line for lines in balanced for line in lines]


def normalize(X):
    X = (X / 255) - 0.5
    return X


def resize(X):
    return K.tf.image.resize_images(X, [66, 200])


def yuv(X):
    return K.tf.image.rgb_to_yuv(X)


model = Sequential()
model.add(Cropping2D(cropping=((58, 24), (0, 0)),
                     input_shape=(160, 320, 3), name='crop'))
model.add(Lambda(resize, name='resize'))
model.add(Lambda(yuv, name='yuv'))
model.add(Lambda(normalize, name='normalize'))
model.add(Conv2D(24, 5, strides=2, activation='elu'))
model.add(Conv2D(36, 5, strides=2, activation='elu'))
model.add(Conv2D(48, 5, strides=2, activation='elu'))
model.add(Conv2D(64, 3, strides=1, activation='elu'))
model.add(Conv2D(64, 3, strides=1, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

lines = helpers.read_lines(sys.argv[1:][0])
lines = balance(lines)
lines = shuffle(lines)
train_samples, valid_samples = train_test_split(lines, test_size=0.2)

print('Training set: {}'.format(len(train_samples)))
print('Validation set: {}'.format(len(valid_samples)))

plt.hist([t[1] for t in train_samples], bins=100)
plt.show()

EPOCHS = 10
BATCH_SIZE = 32

train_generator = helpers.generator(train_samples, BATCH_SIZE)
valid_generator = helpers.generator(valid_samples, BATCH_SIZE)

sample = next(train_generator)

sample_data = sample[0][:10]
sample_labels = sample[1][:10]
fig, ax = plt.subplots(2, 5)
for i in range(0, 5):
    ax[0][i].imshow(sample_data[i])
    ax[0][i].set_title(sample_labels[i])
for i in range(5, 10):
    ax[1][i-5].imshow(sample_data[i])
    ax[1][i-5].set_title(sample_labels[i])
plt.show()

helpers.visualize(model, 'crop', 'resize', sample_data, sample_labels)
helpers.visualize(model, 'yuv', 'yuv', sample_data, sample_labels)

best = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)

history = model\
    .fit_generator(train_generator,
                   steps_per_epoch=int(len(train_samples)/BATCH_SIZE),
                   validation_data=valid_generator,
                   validation_steps=int(len(valid_samples)/BATCH_SIZE),
                   epochs=EPOCHS,
                   verbose=2,
                   callbacks=[best])


plt.plot(history.history['loss'], 'r', label='Training')
plt.plot(history.history['val_loss'], 'b', label='Validation')
plt.legend()

plt.show()
