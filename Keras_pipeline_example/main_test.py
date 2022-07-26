from train_eval_ops import *
import tensorflow as tf
from keras.optimizers import adam
import model_test as mt
import os
from keras.callbacks import ReduceLROnPlateau
from params import *
from python_generator import DataGenerator
from train_eval_ops import *
from keras.layers import Input
import sys
import deeplab3pluss as dl

sys.dont_write_bytecode = True
IDs = [i.replace(".png", "") for i in os.listdir(gen_params["fundus_path"])]

num_train_examples = len(IDs)
num_val_examples = len(IDs)

print("Number of training examples: {}".format(num_train_examples))
print("Number of validation examples: {}".format(num_val_examples))

partition = {'train': IDs, 'validation': IDs}

# Generators
training_generator = DataGenerator(partition['train'], is_training=True, **gen_params)
validation_generator = DataGenerator(partition['validation'], is_training=False, **gen_params)

input_img = Input(params["img_shape"], name='img')

model = mt.get_deep_bunet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
# model = dl.Deeplabv3_transpose(input_tensor=input_img, input_shape=params["img_shape"], training=False, OS=16)

'''Compile model'''
adam = adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=adam, loss=loss_DSSIS_tf11, metrics=[custom_mse, percentual_deviance])
model.summary()

'''train and save model'''
save_model_path = os.path.join(params["save_path"], "weights.hdf5")
cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_percentual_deviance',
                                        save_best_only=True, verbose=1, save_weights_only=True)

learning_rate_reduction = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.000001, verbose=1)

if params["continuing_training"] == True:
    '''Load models trained weights'''
    model.load_weights(save_model_path)

# Train model on dataset
history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              use_multiprocessing=True,
                              steps_per_epoch=int(num_train_examples / (params["batch_size"]) * 10),
                              validation_steps=int(num_train_examples / (params["batch_size"])),
                              epochs=params["epochs"],
                              verbose=1,
                              workers=4,
                              callbacks=[cp, learning_rate_reduction])

pd.DataFrame(history.history).to_csv("loss_curves.csv")
