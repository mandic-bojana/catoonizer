#!/usr/bin/env python

import numpy as np
import cv2 as cv
from os import listdir
from sklearn import preprocessing
from subprocess import call

def save_data(mode):

    x = []
    y = []
   
    images_dir_path = 'reduced_images_' + mode + '/'
    shapes_dir_path = 'reduced_shapes_' + mode + '/' 

    image_files = np.sort(listdir(images_dir_path))
    shape_files = np.sort(listdir(shapes_dir_path))
    
    for image_file, shape_file in zip(image_files, shape_files):
        img = cv.imread(images_dir_path + image_file)
        x.append(img)
        
        shape = np.load(shapes_dir_path + shape_file)
        y.append(shape.ravel())
        
    x = np.array(x)
    x = normalize(x)

    y = np.array(y)
    
    np.save('data_tensors/x_'+ mode, x)
    np.save('data_tensors/y_'+ mode, y)
    
    return 



def normalize(x):
    x = x.astype(np.float32) / 255.0
    return x


def main():

    call(['mkdir', 'data_tensors'])
    
    save_data('train')
    save_data('test')


main()

