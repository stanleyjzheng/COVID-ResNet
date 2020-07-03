import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from loaddataset import processImages

#
#Comment the next 4 lines if you are not using a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

keras.backend.clear_session()
workingDirectory = os.path.dirname(os.path.realpath(__file__))
imgDimensions = 224

#
# image dimensions
# default resnet18 dimensions are 224x224x3 (https://bit.ly/2VaOcyz)
#

img_height = imgDimensions
img_width = imgDimensions
img_channels = 3

def resNet(x):

    def commonLayers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        return y
    
    def groupedConvolution(y, nb_channels, _strides):
        return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)

    def resBlock(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        shortcut = y

        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
        y = commonLayers(y)
        y = groupedConvolution(y, nb_channels_in, _strides=_strides)
        y = commonLayers(y)
        y = layers.BatchNormalization()(y)
        if _project_shortcut or _strides != (1, 1):
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        y = layers.add([shortcut, y])
        y = layers.LeakyReLU()(y)
        return y

    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = commonLayers(x)
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(2):
        project_shortcut = True if i == 0 else False
        strides = (2, 2) if i == 0 else (1, 1)
        x = resBlock(x, 64, 64, _strides=strides, _project_shortcut=project_shortcut)
    for i in range(2):
        strides = (2, 2) if i == 0 else (1, 1)
        x = resBlock(x, 128, 128, _strides=strides)
    for i in range(2):
        strides = (2, 2) if i == 0 else (1, 1)
        x = resBlock(x, 256, 256, _strides=strides)
    for i in range(2):
        strides = (2, 2) if i == 0 else (1, 1)
        x = resBlock(x, 512, 512, _strides=strides)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, activation = 'sigmoid')(x)
    return x

images, labels, verImg, verLabels = processImages(workingDirectory, imgDimensions) #Load from loaddatset.py

metrics = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
network_output = resNet(image_tensor)

model = models.Model(inputs=[image_tensor], outputs=[network_output])

opt = tf.keras.optimizers.SGD(learning_rate = 1, nesterov = True, momentum = 0.9, decay = 0.01)
model.compile(optimizer = opt, loss=keras.losses.BinaryCrossentropy(), metrics = metrics)

model.fit(images, labels, epochs = 35, validation_data = (verImg, verLabels), batch_size = 32)
model.save('workingdirectory' + '/ResNetPretrained')

#print(model.summary())