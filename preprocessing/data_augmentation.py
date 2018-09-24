#!/usr/bin/env python

import numpy as np
import cv2 as cv
from os import listdir
from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage
from numpy import linalg
import sys



def augment_data(images_dir_name, shapes_dir_name):

    
    image_files = np.sort(listdir(images_dir_name))
    shape_files = np.sort(listdir(shapes_dir_name))
    
    for image_file, shape_file in zip(image_files, shape_files):
        
        img = cv.imread(images_dir_name + image_file)
        shape = np.load(shapes_dir_name + shape_file)
        
        mirrored_img, mirrored_shape = mirror_image_and_shape(img, shape)
        rotated_img, rotated_shape = random_rotate_image_and_shape(img.astype('float32'), shape)
        
        cv.imwrite(images_dir_name + image_file[:-3] + '_mirrored.png', mirrored_img)
        np.save(shapes_dir_name + shape_file[:-3] + '_mirrored', mirrored_shape)

        cv.imwrite(images_dir_name + image_file[:-3] + '_rotated.png', rotated_img)
        np.save(shapes_dir_name + shape_file[:-3] + '_rotated', rotated_shape)




def random_rotate_image_and_shape(img, shape):


    h, w, _ = img.shape
    x_center = w / 2
    y_center = h / 2

    facial_key_points_length = len(shape)
    
    sign = np.random.choice([-1, 1])
    theta = sign * np.random.randint(5, 15)
    
    M = cv.getRotationMatrix2D((x_center, y_center), theta, 1)
    rotated_img = cv.warpAffine(img, M, (w, h))

    theta_rads = np.radians(theta)
    rotation_matrix = np.array([
                                  [np.cos(theta_rads), -np.sin(theta_rads)],
                                  [np.sin(theta_rads),  np.cos(theta_rads)]
                               ])
    
    
    rotated_shape = (shape - (x_center, y_center)).dot(rotation_matrix) + (x_center, y_center)
    
    return rotated_img, rotated_shape
    

def mirror_image_and_shape(img, shape):
 
    imgAugmentator = ImageDataGenerator()

    mirrored_img = imgAugmentator.apply_transform(img, transform_parameters={'flip_horizontal' : True})
    img_width = img.shape[1]
    mirrored_shape = swap_mirrored_shape((shape * np.array([-1, 1])) + np.array([img_width, 0]))  
    
    return mirrored_img, mirrored_shape



# When image is mirrored, left eye becomes right eye. Facial landmark points order in shape isn't symmetric, so indices for left and right
# side points of face are manually divided in two arrays in apropriate order for swapping

def swap_mirrored_shape(shape):
    
    left_side_point_indices  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 32, 33, 51, 50, 62, 49, 61, 68, 60, 59, 18, 19, 20, 21, 22,
                                37, 38, 39, 40, 41, 42]) - 1
    
    right_side_point_indices = np.array([17, 16, 15, 14, 13, 12, 11, 10, 36, 35, 53, 54, 64, 55, 65, 66, 56, 57, 27, 26,
                                 25, 24, 23, 46, 45, 44, 43, 48, 47]) - 1
    
    
    
    for left_index, right_index in zip(left_side_point_indices, right_side_point_indices):
        shape[[left_index, right_index], :] = shape[[right_index, left_index], :]
        
    return shape



def main():

    if len(sys.argv) != 2:
        print('usage: ' + sys.argv[0] + ' mode')
        print('(mode can be train or test)')
        return    

    mode = sys.argv[1]

    images_dir_name = 'reduced_images_' + mode + '/'    
    shapes_dir_name = 'reduced_shapes_' + mode + '/'

    print('-- Augmenting data...')
    augment_data(images_dir_name, shapes_dir_name)

    print('-- Augmentation finished.')
    
    
main()
