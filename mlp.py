import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
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

images, labels, verImg, verLabels = processImages(workingDirectory, imgDimensions) #Load from loaddatset.py
    
def buildModel():
    global labels
    global images
    global verImg
    global verLabels
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(227, 227)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1, activation=('sigmoid'))
    ])
    print(model.summary())
    model.compile(optimizer='adadelta',
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])
    model.fit(images, labels, epochs = 2, validation_data = (verImg, verLabels), shuffle = True)

buildModel()