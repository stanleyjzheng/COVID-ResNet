import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
import tensorflow as tf
from tensorflow import keras
import shutil
from cv2 import cv2
from keras import layers
from keras import models

tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

keras.backend.clear_session()
workingDirectory = os.path.dirname(os.path.realpath(__file__))
covidPath = os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'COVID-19'])
normalPath = os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'NORMAL'])
normalImages = list(paths.list_images(f'{normalPath}'))
covidImages = list(paths.list_images(f'{covidPath}'))
images = []
labels = []

def ceildiv(a, b):
    return -(-a // b)

def plotImages(imagePath, figureSize=(10,5), rows=5, titles=None, mainTitle=None, number = 20): #Plots a 5x4 grid of images (not processed)
    f = plt.figure(figsize=figureSize)
    if mainTitle is not None: plt.suptitle(mainTitle, fontsize=50)
    for i in range(0, number):
        sp = f.add_subplot(rows, ceildiv(number, rows), i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        img = plt.imread(imagePath[i])
        plt.imshow(img)

def processImages(normalImages = normalImages, covidImages = covidImages): #converts image directories to resized greyscale numpy arrays
    global images
    global labels
    for i in covidImages:
        label = i.split(os.path.sep)[-2]
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#Turn image into greyscale to save memory
        image = cv2.resize(image,(240, 240)) #Resizes images to 240x240 to save memory. 
        images.append(image)
        labels.append(label)
    for i in normalImages:
        label = i.split(os.path.sep)[-2]
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Turn image into greyscale to save memory. 
        image = cv2.resize(image,(240, 240))
        images.append(image)
        labels.append(label)
    images = np.asarray(images)
    labels = np.asarray(labels)

#
# image dimensions
#

img_height = 240
img_width = 240
img_channels = 3

#
# network params
# change to 1 for ResNet, or 32 for ResNeXt

cardinality = 32


def residual_network(x):

    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))

        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        shortcut = y

        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        if _project_shortcut or _strides != (1, 1):
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])
        y = layers.LeakyReLU()(y)

        return y

    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)

    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)

    for i in range(4):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 512, _strides=strides)

    for i in range(6):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 512, 1024, _strides=strides)

    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 1024, 2048, _strides=strides)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2)(x)

    return x


image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
network_output = residual_network(image_tensor)
model = models.Model(inputs=[image_tensor], outputs=[network_output])
#print(model.summary())


processImages()
labels = [1 if x=='COVID-19' else x for x in labels]
labels = [0 if x=='NORMAL' else x for x in labels]
labels = np.asarray(labels)
images = images / 255.0
model.compile(optimizer='adadelta', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(images, labels, epochs = 3, batch_size = 10)
#expand greyscale to 3 dimension with 1 dimenonal zeroes