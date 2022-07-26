# Read .csv file
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import cv2
from keras.preprocessing.image import img_to_array, load_img

if __name__ == '__main__':

    dataset = pd.read_csv('data/trainLabels_1.csv', sep=',')
    dataset["PatientID"] = ''
    header_list = list(dataset.columns)
    print(header_list)
    patientIDList = []

    for index, row in dataset.iterrows():
        patientID = row[0] + ''
        # patientID = patientID.replace('_right', '')
        # patientID = patientID.replace('_left', '')
        dataset.at[index, 'PatientID'] = patientID
        patientIDList.append(patientID)

    print(dataset.head(5))

    global uniquePatientIDList
    uniquePatientIDList = sorted(set(patientIDList))

    labelList = dataset.loc[:, 'level'].values
    IDList = dataset.loc[:, 'image'].values

    path = "C://Users/Nalu/Documents/ICB/DiabeticRetinopathyDetection/data/train/"
    images = os.listdir(path)
    print("Number of files is " + str(len(images)))

    pd.Series(labelList).value_counts()

    H = 256
    W = 256
    D = 3

    # for imName in images:
    ### for i, id in enumerate(patientIDList):
    # for id in IDList:
        # print("hello")
        # if imName == "data/trainLabels_1.csv":
        #    continue
        #   img = cv2.imread(os.path.join(path, "data/trainLabels_1.csv" + '.jpeg'), 0)
        ### img = cv2.imread(os.path.join(path, id + '.jpeg'), -1)
        ### im_resized = cv2.resize(img, (256, 256))  # .reshape(params["img_shape"])

        ### if i == 2:
            # cv2.imshow('image', img)
            ### plt.imshow(img)  #, cmap='gray', interpolation='bicubic')
            ### plt.show()

    ### print(str(len(img)))
    # cv2.imshow('image', img[2])

        # im_resized = cv2.resize(img, (256, 256))  # .reshape((256, 256, 1))
        # imPath = os.path.join(os.path.sep, path, imName)
        # im = load_img(imPath)
        # arr = img_to_array(im)  # Create numpy array with training images

        # dim1 = arr.shape[0]
        # dim2 = arr.shape[1]
        # dim3 = arr.shape[2]
        # if dim1 < H or dim2 < W or dim3 < D:
        #     print("Error image dimensions are less than expected "+str(arr.shape))
        # arr = cv2.resize(arr, (H, W))  # Numpy array with shape (HEIGHT, WIDTH,3)
        # dim1 = arr.shape[0]
        # dim2 = arr.shape[1]
        # dim3 = arr.shape[2]
        # if dim1 != H or dim2 != W or dim3 != D:
        #    print("Error after resize, image dimensions are not equal to expected "+str(arr.shape))
        # print(type(arr))
        # scale the raw pixel intensities to the range [0, 1] - TBD TEST
        # arr = np.array(arr, dtype="float") / 255.0
        # imName = imName.replace('.jpeg', '')

