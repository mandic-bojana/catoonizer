#!/usr/bin/env python

import numpy as np
import keras.backend as K
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from keras.models import load_model
from patches_creation import prepare_patches_data





def main():


    initial_model = load_model('initial_shape_model/initial_shape_model.hdf5')
    print('-- Initial shape model loaded...\n')
    
    self_updating_cnn_model = load_model('self_updating_cnn/self_updating_CNN.hdf5')
    print('-- Self updating cnn model loaded...\n')

    x_test = np.load('data_tensors/x_test.npy')
    y_current_shape_prediction = initial_model.predict(x_test)    
    y_test = np.load('data_tensors/y_test.npy')
    
   

    for i in range(4):

        self_updating_cnn_inputs = prepare_patches_data(x_test, y_current_shape_prediction)
        y_current_delta_prediction = self_updating_cnn_model.predict(self_updating_cnn_inputs)
        y_current_shape_prediction += y_current_delta_prediction
    
        eval = self_updating_cnn_model.evaluate(self_updating_cnn_inputs, y_test - y_current_shape_prediction)        
        print('evaluation step ' + str(i+1) + ': ', eval)
    
    



main()
