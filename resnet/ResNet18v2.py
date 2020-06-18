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
import datetime

#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)

tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

keras.backend.clear_session()
workingDirectory = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
covidPath = os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'COVID-19'])
normalPath = os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'NORMAL'])
verificationPath = os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'VERIFICATION'])
normalImages = list(paths.list_images(f'{normalPath}'))
covidImages = list(paths.list_images(f'{covidPath}'))
images = []
labels = []
verImg = []
verLabels = []

def processImages(normalImages = normalImages, covidImages = covidImages): #converts image directories to resized greyscale numpy arrays
    global images
    global labels
    global verImg
    global verLabels
    for i in covidImages:
        label = i.split(os.path.sep)[-2]
        image = cv2.imread(i)
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#Turn image into greyscale to save memory
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(227, 227))
        images.append(image)
        labels.append(label)
    for i in normalImages:
        label = i.split(os.path.sep)[-2]
        image = cv2.imread(i)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Turn image into greyscale to save memory. 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(227, 227))
        images.append(image)
        labels.append(label)
    #for i in verificationImages:
    for (index, row) in pd.read_csv(os.path.sep.join([f'{workingDirectory}', 'verification.csv'])).iterrows():
        verLabels.append(row['finding'])
        image = cv2.imread(os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'VERIFICATION', str(row['filename'])]))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Turn image into greyscale to save memory. 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(227, 227))
        verImg.append(image)
    images = np.asarray(images)
    labels = np.asarray(labels)
    verImg = np.asarray(verImg)
    verLabels = np.asarray(verLabels)
#    images = np.expand_dims(images, axis=3)# This line and the line below are only needed for black and white
#    verImg = np.expand_dims(verImg, axis = 3)
    labels = [1 if x=='COVID-19' else x for x in labels]
    labels = [0 if x=='NORMAL' else x for x in labels]
    labels = np.asarray(labels)
    verLabels = [1 if x=='COVID-19' else x for x in verLabels]
    verLabels = [0 if x=='normal' else x for x in verLabels]
    verLabels = np.asarray(verLabels)
    images = images / 255.0
    verImg = verImg / 255.0
#
# image dimensions
# default resnet18 dimensions are 227x227x3 (https://bit.ly/2VaOcyz)
#
img_height = 227
img_width = 227
img_channels = 3

def residual_network(x):

    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)#THIS SHOULD BE UNCOMMENTED BUT IT RUINS THE PARAM COUNT
        return y
    
    def grouped_convolution(y, nb_channels, _strides):
        return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        shortcut = y

        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
#        y = add_common_layers(y)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
#        y = add_common_layers(y)
        #batch normalization is employed after aggregating the transformations and before adding to the shortcut
#        y = layers.BatchNormalization()(y)
        if _project_shortcut or _strides != (1, 1):
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        y = layers.add([shortcut, y])
#        y = layers.LeakyReLU()(y) #THIS SHOULD ALSO BE UNCOMMENTED
        return y

    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
#    x = add_common_layers(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(2):
        project_shortcut = True if i == 0 else False
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 64, 64, _strides=strides, _project_shortcut=project_shortcut)
    for i in range(2):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 128, 128, _strides=strides)
    for i in range(2):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 256, _strides=strides)
    for i in range(2):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 512, 512, _strides=strides)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000, activation = 'softmax')(x)
    x = layers.Dense(2)(x)
    return x

processImages()
image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
network_output = residual_network(image_tensor)
model = models.Model(inputs=[image_tensor], outputs=[network_output])
print(model.summary())
model.compile(optimizer = 'SGD', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

history = model.fit(images, labels, epochs = 3, validation_data = (verImg, verLabels), shuffle = True)#, callbacks = [tensorboard_callback])