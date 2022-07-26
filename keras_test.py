# import packages
#https://www.kaggle.com/meenavyas/diabetic-retinopathy-detection/data

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import keras

code = 1
NumClass = 5
W = 128
H = 128
D = 3
InputShape = (H, W, D)
BatchSize = 32
Epochs = 1e-3
#global variables
ImageNameDataHash = {}
uniquePatientIDList = []


def class_to_int(label):
    if label == "Healthy":
        return 0
    if label == "Mild":
        return 1
    if label == "Moderate":
        return 2
    if label == "Severe":
        return 3
    if label == "Proliferative":
        return 4
    print("ERROR: Invalid label", label)
    return 5


def int_to_class(idx):
    if idx == 0:
        return "Healthy"
    if idx == 1:
        return "Mild"
    if idx == 2:
        return "Moderate"
    if idx == 3:
        return "Severe"
    if idx == 4:
        return "Proliferative"
    print("ERROR: Invalid class", idx)
    return 5


def readTrainingData(path):

    if code == 1:
        global ImageNameDataHash
        images = os.listdir(path)
        print("Number of files is" + str(len(images)))
        for imName in images:
            if imName == "trainLabels.csv":
                continue
            imPath = os.path.join(os.path.sep, path, imName)
            im = load_img(imPath)
            arr = img_to_array(im) #numpy array with shape (233,233,3)
            dim1 = arr.shape[0]
            dim2 = arr.shape[1]
            dim3 = arr.shape[2]
            if (dim1 != H or dim2 != W or dim3 != D):
                print("Error after resize, image dimensions are not equal to expected "+str(arr.shape))
            arr = np.array(arr, dtype="float") / 255.0
            imName = imName.replace('.jpeg','')
            ImageNameDataHash[str(imName)] = np.array(arr)
            return


#READ TRAINING DATA
readTrainingData("/data/sample")


def readTrainingCsv():
    if code == 1:
        labels = pd.read_csv('data/sampleLabels.csv', sep=',')
        rowCount = labels.shape[0]
        colCount = labels.shape[1]
        labels["PatientID"] = ''
        headerList = list(labels.columns)
        patientIDList = []
        for i, row in labels.iterrows():
            key = row[0]
            patientID = row[0] + ''
            patientID = patientID.replace('_right','')
            patientID = patientID.replace('_left','')
            labels.at[index, 'PatientID'] = patient

    elif code == 2:
        #dataset = np.loadtxt("sampleLabels.csv", delimiter=",")
        #image = dataset[:,1]
        #level = dataset[:,2]
        dataset = pd.read_csv('data/sampleLabels.csv', sep=',', skiprows =1 )



def preProcessing():
#1. rescale to same radius (300 pixels or 500 pixels)
#2. subtracted local average color; the local average gets mapped to 50% gray
#3. clipped the images to 90% size (remove boundary)


def createModel():
    model = Sequential()

    model.add(ResNet50(pooling='avg', weights=restnet_weights_path))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(NumClass, activation='softmax'))

    model.layers[0].trainable = False

    opt = Adam()
    model.compile(optimizer=opt, loss="categorical_crossentropy"
                  , metrics=['accuracy'])
    return model

