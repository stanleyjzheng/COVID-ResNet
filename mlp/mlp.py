import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
import tensorflow as tf
from tensorflow import keras
import shutil
from cv2 import cv2

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#Turn image into greyscale to save memory
        image = cv2.resize(image,(360, 360)) #Resizes images to 240x240 to save memory. 
        images.append(image)
        labels.append(label)
    for i in normalImages:
        label = i.split(os.path.sep)[-2]
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Turn image into greyscale to save memory. 
        image = cv2.resize(image,(360, 360))
        images.append(image)
        labels.append(label)
    #for i in verificationImages:
    for (index, row) in pd.read_csv(os.path.sep.join([f'{workingDirectory}', 'verification.csv'])).iterrows():
        verLabels.append(row['finding'])
        image = cv2.imread(os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'VERIFICATION', str(row['filename'])]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Turn image into greyscale to save memory. 
        image = cv2.resize(image,(360, 360))
        verImg.append(image)
    images = np.asarray(images)
    labels = np.asarray(labels)
    verImg = np.asarray(verImg)
    verLabels = np.asarray(verLabels)

def buildModel():
    global labels
    global images
    global verImg
    global verLabels
    labels = [1 if x=='COVID-19' else x for x in labels]
    labels = [0 if x=='NORMAL' else x for x in labels]
    labels = np.asarray(labels)
    verLabels = [1 if x=='COVID-19' else x for x in verLabels]
    verLabels = [0 if x=='normal' else x for x in verLabels]
    verLabels = np.asarray(verLabels)
    images = images / 255.0
    verImg = verImg / 255.0


    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(360, 360)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(2, activation=('softmax'))#Make sure to remove activation function if using from_logits = True, otherwise results will be scattered.
    ])
#Relu tanh
    model.compile(optimizer='adadelta',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    
    history = model.fit(images, labels, epochs = 3, validation_data = (verImg, verLabels), shuffle = True)
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='verification')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='verification')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    processImages()
    buildModel()
