import numpy as np
import keras
import cv2
import os
import random
import skimage as sk
from skimage import transform
import scipy
from params import *

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, image_path, label_path, is_training, batch_size=2, dim=gen_params["dim"],
                 n_channels=1, shuffle=True):
        'Initialization'
        self.shape = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.image_path = image_path
        self.label_path = label_path
        self.is_training = is_training

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(int(len(self.list_IDs)))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.shape[0], self.shape[1], self.n_channels))
        y = np.empty((self.batch_size, self.shape[0], self.shape[1], self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #load samples
            im = cv2.imread(os.path.join(self.fundus_path, ID + '.png'),0)
            lbl = np.load(os.path.join(self.thickness_path, ID + ".npy"))
            #resize samples
            im_resized = cv2.resize(im, self.shape).reshape(params["img_shape"])
            lbl_resized = cv2.resize(lbl, self.shape).reshape(params["img_shape"])
            # Store sample
            X[i,] = im_resized
            y[i,] = lbl_resized

            X[i,], y[i,] = self.__pre_process(X[i,],y[i,])

        return X, y

    def __pre_process(self, train_im, label_im):
        # remove non labeled part of image
        train_im[label_im == 0] = 0
        label_im[label_im == 0] = 0

        # remove non labeled part of label
        train_im[train_im == 0] = 0
        label_im[train_im == 0] = 1 #set to one for log operations

        # scaling
        label_im = np.divide(label_im, 500., dtype=np.float32)
        train_im = np.divide(train_im, 255., dtype=np.float32)
        # set all nans to zero
        label_im = np.nan_to_num(label_im)
        train_im = np.nan_to_num(train_im)

        if self.is_training:
            self.augment(train_im,label_im)

        return (train_im.reshape(params["img_shape"]), label_im.reshape(params["img_shape"]))

    def augment(self, train_im, label_im):
        # get boolean if rotate
        rotate = bool(random.getrandbits(1))
        flip_hor = bool(random.getrandbits(1))
        flip_ver = bool(random.getrandbits(1))
        shift = bool(random.getrandbits(1))

        if (rotate):
            train_im, label_im = self.random_rotation(train_im,label_im)
        if (flip_hor):
            train_im, label_im = self.flip_horizontally(train_im, label_im)
        if (flip_ver):
            train_im, label_im = self.flip_vertically(train_im, label_im)
        if (shift):
            train_im, label_im = self.random_shift(train_im, label_im)

        return train_im, label_im

    def random_rotation(self,image_array, label_array):
        # pick a random degree of rotation between 25% on the left and 25% on the right
        random_degree = random.uniform(-25, 25)
        r_im = sk.transform.rotate(image_array, random_degree, preserve_range=True)
        r_l = sk.transform.rotate(label_array, random_degree, preserve_range=True)
        return r_im.astype(np.float32), r_l.astype(np.float32)

    def flip_vertically(self,image_array, label_array):
        flipped_image = np.fliplr(image_array)
        flipped_label = np.fliplr(label_array)
        return (flipped_image, flipped_label)

    def flip_horizontally(self,image_array, label_array):
        flipped_image = np.flipud(image_array)
        flipped_label = np.flipud(label_array)
        return (flipped_image, flipped_label)

    def random_shift(self,image_array, label_array):
        rand_x = random.uniform(-15, 15)
        rand_y = random.uniform(-15, 15)
        image_array = scipy.ndimage.shift(image_array[:, :, 0], (rand_x, rand_y))
        label_array = scipy.ndimage.shift(label_array[:, :, 0], (rand_x, rand_y))
        return (image_array.reshape(self.shape[0], self.shape[1], 1),
                label_array.reshape(self.shape[0], self.shape[1], 1))
