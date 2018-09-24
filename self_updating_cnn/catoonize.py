#!/usr/bin/env python

import cv2 as cv
import sys
from subprocess import call
import numpy as np
from keras.models import load_model
from patches_creation import create_patches, create_dict_input
from prepare_uploaded_image import *
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt




def main():

    if(len(sys.argv) != 2):
        print('usage: ' + sys.argv[0] + ' frontal_face_image_path')
        return 

    
    img_path = sys.argv[1]

    img = cv.imread(img_path)
    predicted_shape = predict_on_single_image(img)

    show_result(img, predicted_shape)

    call(['./exec.sh'])

    
    plt.imshow(cv.imread('head_with_body.png'))

    
    call(['rm', 'coordinates.txt', 'head_with_body.png', 'eyes.png', 'face.png', 'head_with_ears.png'])





def predict_on_single_image(img):
        
        initial_model = load_model('../initial_shape_model/initial_shape_model.hdf5') 
        self_updating_cnn_model = load_model('self_updating_CNN.hdf5')



        reduced_img = detect_crop_and_rescale_face(img)
        

        if(len(reduced_img) == 0):
            print('none')
            return    
    
        standardized_img = reduced_img.astype('float32') / 255.0

        y_current_shape_prediction = initial_model.predict(standardized_img.reshape(1, 320, 240, 3))

        for i in range(4):
            patches = create_patches(standardized_img, y_current_shape_prediction.reshape(68, 2))
            self_updating_cnn_inputs = create_dict_input(patches.reshape(1, 68, 45, 45, 3))
            y_current_delta_prediction = self_updating_cnn_model.predict(self_updating_cnn_inputs)
            y_current_shape_prediction += y_current_delta_prediction

        write_coordinates(y_current_shape_prediction)
        cv.imwrite('face_image.png', reduced_img)
        
        
        show_result(reduced_img, y_current_shape_prediction)

        return y_current_shape_prediction



def show_result(img, shape):
    plt.imshow(img.astype(np.float32))
    plt.scatter(shape.reshape(68, 2)[:, 0], shape.reshape(68, 2)[:, 1], color='green', s=3)  
    plt.show()



main()
