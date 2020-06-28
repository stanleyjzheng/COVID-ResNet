from cv2 import cv2
import numpy as np
from imutils import paths
import os
import pandas as pd


def processImages(workingDirectory, imageDimensions):
    images = []
    labels = []
    verImg = []
    verLabels = []
    covidPath = os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'COVID-19'])
    normalPath = os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'NORMAL'])
    verificationPath = os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'VERIFICATION'])
    normalImages = list(paths.list_images(f'{normalPath}'))
    covidImages = list(paths.list_images(f'{covidPath}'))
    for i in covidImages:
        label = i.split(os.path.sep)[-2]
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(imageDimensions, imageDimensions))
        images.append(image)
        labels.append(label)
    print('Finished copying COVID-19 images')
    for i in normalImages:
        label = i.split(os.path.sep)[-2]
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(imageDimensions, imageDimensions))
        images.append(image)
        labels.append(label)
    print('Finished copying normal images')
    for (index, row) in pd.read_csv(os.path.sep.join([f'{workingDirectory}', 'verification.csv'])).iterrows():
        verLabels.append(row['finding'])
        image = cv2.imread(os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'VERIFICATION', str(row['filename'])]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(imageDimensions, imageDimensions))
        verImg.append(image)
    print('Finished copying verification images')
    images = np.asarray(images)
    labels = np.asarray(labels)
    verImg = np.asarray(verImg)
    verLabels = np.asarray(verLabels)
    labels = [1 if x=='COVID-19' else x for x in labels]
    labels = [0 if x=='NORMAL' else x for x in labels]
    labels = np.asarray(labels)
    verLabels = [1 if x=='COVID-19' else x for x in verLabels]
    verLabels = [0 if x=='normal' else x for x in verLabels]
    verLabels = np.asarray(verLabels)
    images = images / 255.0
    verImg = verImg / 255.0
    print('Number of COVID train files:',str(len(covidImages)))
    print('Number of normal train files',str(len(normalImages)))
    print('Number of verification images', str(len(list(paths.list_images(f'{verificationPath}')))))
    return images, labels, verImg, verLabels