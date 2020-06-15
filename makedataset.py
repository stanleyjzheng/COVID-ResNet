import os
from imutils import paths
import shutil
import pandas as pd
'''please see createdataset.md before running this.'''
'''Before running this, you must create a directory called "data". Inside data must be 3 folders, one called "COVID-19", another called "NORMAL", a third called "COVIDVERIFICATION", and a fourth named "NORMALVERIFICATION"'''

workingDirectory = os.path.dirname(os.path.realpath(__file__)) #This assumes your datsets were cloned into the same directory as this makedataset.py file. Change this to the directory of your datasets if this is not the case
covidPath = os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'COVID-19'])# Uses COVID-19 Radiography Database as root folder. Should probably change this to data/covid19, etc. 
normalPath = os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'NORMAL'])#
testPath = os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'VERIFICATION'])

def makeDataset(workingDirectory = workingDirectory): #Combines all COVID images into one singular dataset to make it easier to process
    print('starting, please be patient. This can take a few minutes. ')
    global covidPath
    global normalPath
    imagePath = os.path.sep.join([f'{workingDirectory}', 'covid-chestxray-dataset', 'images'])
    fig1ImagePath =  os.path.sep.join([f'{workingDirectory}', 'Figure1-COVID-chestxray-dataset', 'images'])
    actualmedImagePath = os.path.sep.join([f'{workingDirectory}', 'Actualmed-COVID-chestxray-dataset', 'images'])
    for (index, row) in pd.read_csv(os.path.sep.join([f'{workingDirectory}', 'covid-chestxray-dataset', 'metadata.csv'])).iterrows():#Add covid-chestxray-dataset and metadata.csv to current working directory
        if row['finding'] != 'COVID-19':#Skip everything except if the case is COVID-19 
            continue
        singleImagePath = os.path.sep.join([imagePath, row['filename']])
        if not os.path.exists(singleImagePath):
            continue
        filename = row['filename'].split(os.path.sep)[-1]
        outputPath = os.path.sep.join([f'{covidPath}', filename])
        shutil.copy2(singleImagePath, outputPath)
    for (index, row) in pd.read_csv(os.path.sep.join([f'{workingDirectory}', 'Figure1-COVID-chestxray-dataset', 'metadata.csv']), encoding = 'unicode_escape').iterrows():
        if row['finding'] != 'COVID-19':
            continue
        if os.path.exists(os.path.join(f'{fig1ImagePath}', 'row[patientid]'+'.jpg')):
            singleImagePath = os.path.sep.join([fig1ImagePath, 'row[patientid]' + '.jpg'])
        elif os.path.exists(os.path.join(f'{fig1ImagePath}', 'row[patientid]'+'.png')):
            singleImagePath = os.path.sep.join([fig1ImagePath, 'row[patientid]' + '.png'])
        else:
            continue
        filename = singleImagePath.split(os.path.sep)[-1]#FIND FILENAME FROM patientid
        outputPath = os.path.sep.join([f'{covidPath}', filename])
        shutil.copy2(singleImagePath, outputPath)
    for (index, row) in pd.read_csv(os.path.sep.join([f'{workingDirectory}', 'Actualmed-COVID-chestxray-dataset', 'metadata.csv']), encoding = 'unicode_escape').iterrows():
        if row ['finding'] != 'No finding':
            continue
        singleImagePath = os.path.sep.join([actualmedImagePath, row['imagename']])
        if not os.path.exists(singleImagePath):
            continue
        filename = row['imagename'].split(os.path.sep)[-1]
        outputPath = os.path.sep.join([f'{normalPath}', filename])
        shutil.copy2(singleImagePath, outputPath)
    for (index, row) in pd.read_csv(os.path.sep.join([f'{workingDirectory}', 'Actualmed-COVID-chestxray-dataset', 'metadata.csv']), encoding = 'unicode_escape').iterrows():
        if row ['finding'] != 'COVID-19':
            continue
        singleImagePath = os.path.sep.join([actualmedImagePath, row['imagename']])
        if not os.path.exists(singleImagePath):
            continue
        filename = row['imagename'].split(os.path.sep)[-1]
        outputPath = os.path.sep.join([f'{covidPath}', filename])
        shutil.copy2(singleImagePath, outputPath)
    print('finished copying.')
    print('moving verification files')
    normalImages = list(paths.list_images(f'{normalPath}'))
    covidImages = list(paths.list_images(f'{covidPath}'))
    #Use shutil to shutil.move(source, destination) if it is in the verification file. 
    print('dataset completed. you may now close this script and begin to train.')

makeDataset()