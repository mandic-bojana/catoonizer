#!/usr/bin/env python


import numpy as np
import cv2 as cv
from os import listdir
from sklearn import preprocessing
from sklearn import model_selection
from keras import losses
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import sys
from model import create_model


def main():

    if len(sys.argv) != 4:
        print('usage: ' + sys.argv[0] + ' epochs' + ' batch_size' + ' learining_rate')
        return  
    
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    lr = float(sys.argv[3])

    x_train_val = np.load('data_tensors/x_train.npy')  
    y_train_val = np.load('data_tensors/y_train.npy')

    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train_val, y_train_val,
                                                                            test_size=0.1, random_state=3)


    model = create_model()
    print('-- Model created...')

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print('-- Model compiled...')  

    checkpoint = ModelCheckpoint('callback_history/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)

    callbacks_list = [checkpoint]

    K.set_value(model.optimizer.lr, lr)

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks_list)

    model.save('initial_shape_model.hdf5')
    print('-- Model saved.')


    del model


main()
