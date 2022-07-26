# import packages
# https://www.kaggle.com/meenavyas/diabetic-retinopathy-detection/data

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import keras


class DataGenerator:

    def __init__(self, label_path, image_path):
        self.label_path = label_path
        self.image_path = image_path

    def read_csv(self):
        data = pd.read_csv(self.label_path, sep=',')

        # Add PatientID column (number before _left/_right)
        data["PatientID"] = ''
        patientIDList = []
        for index, row in data.iterrows():
            patientID = row[0] + ''
            patientID = patientID.replace('_right', '')
            patientID = patientID.replace('_left', '')
            data.at[index, 'PatientID'] = patientID
            patientIDList.append(patientID)

        global uniquePatientIDList
        uniquePatientIDList = sorted(set(patientIDList))

        return data

    def generate_data(self, ):
        images = os.listdir(self.image_path)
        for imName in images:
            # load images
            if imName == "data/trainLabels.csv":
                continue
            imPath = os.path.join(os.path.sep, self.image_path, imName)
            im = load_img(imPath)
            # Create numpy array with training images
            arr = img_to_array(im)

            # resize images
            dim1 = arr.shape[0]
            dim2 = arr.shape[1]
            dim3 = arr.shape[2]
            if dim1 != H or dim2 != W or dim3 != D:
                print("Error after resize, image dimensions are not equal to expected "+str(arr.shape))
            arr = cv2.resize(arr, (H, W)) # Numpy array with shape (HEIGHT, WIDTH,3)
            dim1 = arr.shape[0]
            dim2 = arr.shape[1]
            dim3 = arr.shape[2]
            if dim1 != H or dim2 != W or dim3 != D:
                print("Error after resize, image dimensions are not equal to expected "+str(arr.shape))
            # print(type(arr))
            # scale the raw pixel intensities to the range [0, 1] - TBD TEST
            arr = np.array(arr, dtype="float") / 255.0
            imName = imName.replace('.jpeg', '')

        X = np.empty((self.batch_size, self.shape[0],self.shape[1], self.n_channels))
        y = np.empty((self.batch_size, self.shape[0],self.shape[1], self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load samples
            im = cv2.imread(os.path.join(self.fundus_path, ID + '.png'), 0)
            lbl = np.load(os.path.join(self.thickness_path, ID + ".npy"))
            # resize samples
            im_resized = cv2.resize(im, self.shape).reshape(params["img_shape"])
            lbl_resized = cv2.resize(lbl, self.shape).reshape(params["img_shape"])
            # Store sample
            X[i,] = im_resized
            y[i,] = lbl_resized

            X[i,], y[i,] = self.__pre_process(X[i,],y[i,])

        return X, y

