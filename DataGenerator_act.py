# Read .csv file
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import cv2
import skimage as sk
from skimage import transform
import scipy
from keras.preprocessing.image import img_to_array, load_img


class DataGenerator:

    def __init__(self, image_path, label_path, is_training, batch_size=2, dim=gen_params["dim"],
                 n_channels=1, shuffle=True):
        self.label_path = label_path
        self.image_path = image_path
        self.shape = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.image_path = image_path
        self.label_path = label_path
        self.is_training = is_training

    def generate_data(self, ):
        X = np.empty((self.batch_size, self.shape[0], self.shape[1], self.n_channels))
        y = np.empty((self.batch_size, self.shape[0], self.shape[1], self.n_channels))

        # read .csv file
        dataset = pd.read_csv(self.label_path, sep=',')
        # get numpy array of dataset columns
        labelList = dataset.loc[:, 'level'].values
        IDList = dataset.loc[:, 'image'].values

        for i, ID in enumerate(IDList):
            # read .jpeg images
            img = cv2.imread(os.path.join(self.image_path, ID + '.jpeg'), 1)
            # resize images
            im_resized = cv2.resize(img, self.shape).reshape(params["img_shape"])

            # plot one image and its resized version
            if i == 2:
                fig = plt.figure()
                plt.subplot(211)
                plt.imshow(img)
                plt.subplot(212)
                plt.imshow(im_resize)
                plt.show()

            X[i, ] = im_resized
            y[i, ] = labelList
            X[i, ], y[i, ] = self.pre_process(X[i, ], y[i, ])

        return X, y

    def pre_process(self, train_im, label_im):
        # scaling
        label_im = np.divide(label_im, 500., dtype=np.float32)
        train_im = np.divide(train_im, 255., dtype=np.float32)

        if self.is_training:
            self.balance_distribution(train_im, label_im)
            self.crop(train_im)
            self.augment_space(train_im, label_im)
            self.augment_color(train_im)

        train_im = train_im.reshape(params["img_shape"])
        label_im = label_im.reshape(params["img_shape"])

        return train_im, label_im

    # Pre process scale function of btgraham!!!!!!!!!!!!!!!!?
    def scaleRadius(self, im, scale):
        x = im[im.shape[0]/2, :, :].sum(1)
        r = (x > x.mean()/10).sum()/2
        s = scale*1.0/r
        return cv2.resize(im, (0, 0), fx=s, fy=s)

    def balance_distribution(self, train_im, label_im):
        if len(class) <

    def crop(self, train_im):

    def augment_space(self, train_im, label_im):
        # get boolean if rotate
        rotate = bool(random.getrandbits(1))
        flip_hor = bool(random.getrandbits(1))
        flip_ver = bool(random.getrandbits(1))
        shift = bool(random.getrandbits(1))

        if rotate:
            train_im, label_im = self.random_rotation(train_im,label_im)
        if flip_hor:
            train_im, label_im = self.flip_horizontally(train_im, label_im)
        if flip_ver:
            train_im, label_im = self.flip_vertically(train_im, label_im)
        if shift:
            train_im, label_im = self.random_shift(train_im, label_im)

        return train_im, label_im

    def random_rotation(self, image_array, label_array):
        # pick a random degree of rotation between 25% on the left and 25% on the right
        random_degree = random.uniform(-25, 25)
        r_im = sk.transform.rotate(image_array, random_degree, preserve_range=True)
        r_l = sk.transform.rotate(label_array, random_degree, preserve_range=True)
        return r_im.astype(np.float32), r_l.astype(np.float32)

    def flip_vertically(self, image_array, label_array):
        flipped_image = np.fliplr(image_array)
        flipped_label = np.fliplr(label_array)
        # X = tf.image.random_flip_up_down(X)
        return flipped_image, flipped_label

    def flip_horizontally(self, image_array, label_array):
        flipped_image = np.flipud(image_array)
        flipped_label = np.flipud(label_array)
        # X = tf.image.random_flip_left_right(X)
        return flipped_image, flipped_label

    def random_shift(self, image_array, label_array):
        rand_x = random.uniform(-15, 15)
        rand_y = random.uniform(-15, 15)
        image_array = scipy.ndimage.shift(image_array[:, :, 0], (rand_x, rand_y))
        label_array = scipy.ndimage.shift(label_array[:, :, 0], (rand_x, rand_y))
        return (image_array.reshape(self.shape[0], self.shape[1], 1),
                label_array.reshape(self.shape[0], self.shape[1], 1))

    def augment_color(self, train_im):
        # get boolean if rotate
        brightness = bool(random.getrandbits(1))
        contrast = bool(random.getrandbits(1))
        saturation = bool(random.getrandbits(1))

        if brightness:
            train_im, label_im = self.random_brightness(train_im)
        if contrast:
            train_im, label_im = self.random_contrast(train_im)
        if saturation:
            train_im, label_im = self.random_saturation(train_im)

        return train_im

    def random_brightness(self, image_array):
        mod_image = tf.image.random_brightness(image_array, max_delta=0.1)
        return mod_image

    def random_saturation(self, image_array):
        mod_image = tf.image.random_saturation(image_array, lower=0.75, upper=1.5)
        return mod_image

    def random_contrast(self, image_array):
        mod_image = tf.image.random_contrast(image_array, lower=0.75, upper=1.5)
        return mod_image
