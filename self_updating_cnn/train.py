#!/usr/bin/env python


import numpy as np
import sys
import keras.backend as K
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from keras.models import load_model
from model import create_self_updating_CNN_model
from patches_creation import prepare_patches_data
from sklearn import model_selection



            

def main():

    if len(sys.argv) != 5:
        print('usage: ' + sys.argv[0] + ' iterations' + ' epochs' + ' batch_size' + ' learining_rate')
        return  
    
    iterations = int(sys.argv[1])
    epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    lr = float(sys.argv[4]) 


    initial_shape_model = load_model('initial_shape_model/initial_shape_model.hdf5')
    print('-- Initial model loaded...\n')

    self_updating_CNN = create_self_updating_CNN_model()
    print('-- Self-updating CNN model created...\n')

    self_updating_CNN.compile(optimizer='adam', loss='mse', metrics=['mae'])

    checkpoint = ModelCheckpoint('callback_history/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)

    epochs = epochs
    decay_rate = 1.0 / epochs
    schedule = lambda epoch_index, lr: lr * (1. / (1. + decay_rate * epoch_index))
    lrScheduler = LearningRateScheduler(schedule, verbose=1)

    K.set_value(self_updating_CNN.optimizer.lr, lr)

    callbacks_list = [checkpoint, lrScheduler]



 
    x_train_val = np.load('data_tensors/x_train.npy')  
    y_train_val = np.load('data_tensors/y_train.npy')

    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train_val, y_train_val,
                                                                            test_size=0.1, random_state=3)
  
    y_train_predicted = initial_shape_model.predict(x_train)
    y_val_predicted = initial_shape_model.predict(x_val)   

    train_inputs = prepare_patches_data(x_train, y_train_predicted)
    val_inputs = prepare_patches_data(x_val, y_val_predicted)

    y_delta_val = y_val - y_val_predicted
    val_outputs = {'output': y_delta_val } 

    print(train_inputs['input_0'].shape)
    print(val_inputs['input_0'].shape)

    for i in range(iterations):

        y_sampled = create_samples(y_train_predicted, y_train)
        y_delta_train = y_train - y_sampled
        
        train_outputs = {'output': y_delta_train }        
        self_updating_CNN.fit(train_inputs, train_outputs, epochs=epochs, batch_size=batch_size, 
                                            validation_data = (val_inputs, val_outputs), callbacks=callbacks_list)

  
    self_updating_CNN.save('self_updating_CNN.hdf5')



def create_samples(predicted_shapes, true_shapes):

    sample_shapes = []

    facial_landmark_points_length = predicted_shapes.shape[1] / 2

    for i in range(len(predicted_shapes)):
        choosen_shapes = np.random.choice(['predicted', 'true'])
    
        if choosen_shapes=='predicted':
            new_sample = predicted_shapes[i]
        else:
            new_sample = true_shapes[i]
    
            
    # rotation angle (-0.12, 0.12)
        theta = np.random.uniform(low=-0.12, high=0.12)
    # translation vector (5)
        d = 5 * np.random.randn(2)
    # scaling factor (different scale of x and y) (0.93, 1.07)
        f = np.random.uniform(low=0.93, high=1.07, size=2)
    
        rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
        ])
    
    # applying transformation
        new_sample = (new_sample.reshape(facial_landmark_points_length, 2) * f).dot(rotation_matrix) + d
        sample_shapes.append(new_sample)
        
    sample_shapes = np.array(sample_shapes)
    
    return sample_shapes.reshape(-1, facial_landmark_points_length * 2)
    

main()
    
