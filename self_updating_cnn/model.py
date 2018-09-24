from keras import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, concatenate, Activation, BatchNormalization, Flatten, Dense, ZeroPadding2D
import tensorflow as tf
import numpy as np
import keras.backend as K


def create_self_updating_CNN_model(facial_landmark_points_length=68):
    
    with tf.variable_scope('suCNN_model'):
        patches_inputs = []
    
        for i in range(facial_landmark_points_length):
            single_patch_input = Input(shape=(45, 45, 3), name='input_'+str(i))
            patches_inputs.append(single_patch_input)
        
        patches_outputs = []   
    
        for i in range(facial_landmark_points_length):                
            single_patch_output = Conv2D(input_shape=(45 , 45, 3), filters=16, kernel_size=(6, 6), strides=(2, 2), 
                                         padding='same',
                                         name='conv1_'+str(i))(patches_inputs[i])
            single_patch_output = BatchNormalization(name="bn1_"+str(i))(single_patch_output)
            single_patch_output = Activation('relu', name='relu1_'+str(i))(single_patch_output)
            single_patch_output = MaxPool2D(pool_size=(2,2), strides=2, 
                                            name='mp1_'+str(i))(single_patch_output)
        
            single_patch_output = Conv2D(filters=16, kernel_size=(3, 3), padding='same',
                                         name='conv2_'+str(i))(single_patch_output)
            single_patch_output = BatchNormalization(name="bn2_"+str(i))(single_patch_output)
            single_patch_output = Activation('relu', name='relu2_'+str(i))(single_patch_output)
            single_patch_output = MaxPool2D(pool_size=(2,2), strides=2, name='mp2_'+str(i))(single_patch_output)
            single_patch_output = Flatten(input_shape=(4, 4, 16), data_format='channels_last',
                                          name='flatt_'+str(i))(single_patch_output)

            
            patches_outputs.append(single_patch_output)
    
    
        concatenated_patches = concatenate(patches_outputs, name='concat')
        output = Dense(units=130, name='dense1')(concatenated_patches)
        output = Dropout(0.2)(output)
        output = BatchNormalization(name='bn3')(output)
        output = Activation('relu', name='relu3')(output)
    
        output = Dense(units=facial_landmark_points_length * 2, name='output')(output)

    
        model = Model(inputs=patches_inputs, outputs=output)
    
        return model

