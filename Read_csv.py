# Read .csv file
import numpy
import pandas as pd
import os


if __name__ == '__main__':

    dataset = pd.read_csv('data/trainLabels.csv', sep=',')
    print(dataset.head(5))

    print(type(dataset))
    row_count = dataset.shape[0]
    col_count = dataset.shape[1]
    print("row_count="+str(row_count)+" col count="+str(col_count))
    dataset["PatientID"] = ''
    header_list = list(dataset.columns)
    print(header_list)

    for index, row in dataset.iterrows():
        patientID = row[0] + ''
        patientID = patientID.replace('_right', '')
        patientID = patientID.replace('_left', '')
        dataset.at[index, 'PatientID'] = patientID
    patient = dataset.loc[:, "image"].values
    print(dataset.head(5))
    print(patient[1:5])

    path = 'data/train'
    images = os.listdir(path)
    print("Number of files is " + str(len(images)))
    for imName in images:
        if imName == "trainLabels.csv":
            continue
        imPath = os.path.join(os.path.sep, path, imName)

