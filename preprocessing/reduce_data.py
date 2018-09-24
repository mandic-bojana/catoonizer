#!/usr/bin/env python

from os import listdir
import re
import numpy as np
import cv2 as cv
import sys
from subprocess import call



def read_data(data_dir_path):

    images = []
    shapes = []
 
    point_regex = re.compile('(-?\d+.\d*) (-?\d+.\d*)')
    
    file_names = np.sort(listdir(data_dir_path))

    for file_name in file_names:
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            image = cv.imread(data_dir_path + '/' + file_name)
            images.append(image)
        else:
            annot_file = open(data_dir_path + '/' + file_name)
            file_lines = annot_file.readlines()
            
            shape = []

            for line in file_lines[3:-1]:
                x_coord = float(re.match(point_regex, line).group(1))
                y_coord = float(re.match(point_regex, line).group(2))
                point = np.array([x_coord, y_coord])
                shape.append(point)
            
            shape = np.array(shape)
            shapes.append(shape)
           
    
    return images, shapes
        


def rect_area(rect_points):
    _, _, h, w = rect_points
    return h * w




def crop_face_and_shape(img, shape, height_over_width_ratio=1.0*320 / 240):


    face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_detector.load('utils/haarcascade_frontalface_default.xml')
    detection = face_detector.detectMultiScale(img, minNeighbors=10, scaleFactor=1.1)
    
    if len(detection) < 1:
        return np.empty(0), np.empty(0) 
    
    # if more then one faces are detected, select the one with maximal area
    max_detected_rec = detection[0]

    for i in range(len(detection)):
        if rect_area(detection[i]) > rect_area(max_detected_rec):
            max_detected_rec = detection[i]
    
    x_rect, y_rect, w_rect, h_rect = max_detected_rec

    # making sure shape points are in cropped picture
    x_max = np.max(shape[:, 0])
    x_min = np.min(shape[:, 0])
    y_max = np.max(shape[:, 1])
    y_min = np.min(shape[:, 1])
     
    # add random border
    border = np.random.randint(0, 30)
    
    x_rect = int(min(x_rect, x_min)) + border
    y_rect = int(min(y_rect, y_min)) + border
    w_rect = int(max(w_rect, x_max - x_rect)) + border
    h_rect = int(max(h_rect, y_max - y_rect)) + border
    

    # Face detector can return bounding box not fully contained in image
    if y_rect < 0:
        img = cv.copyMakeBorder(img, y_rect, 0, 0, 0, cv.BORDER_CONSTANT)
        shape = shape + np.array([0, y_rect])
    if y_rect + h_rect > img.shape[0]:
        img = cv.copyMakeBorder(img, 0, y_rect + h_rect - img.shape[0], 0, 0, cv.BORDER_CONSTANT)
    if x_rect < 0:
        img = cv.copyMakeBorder(img, 0, 0, x_rect, 0, cv.BORDER_CONSTANT)
        shape = shape + np.array([x_rect, 0])
    if x_rect + w_rect > img.shape[1]:
        img = cv.copyMakeBorder(img, 0, 0, 0, x_rect + w_rect - img.shape[1], cv.BORDER_CONSTANT)
    
    
    # update height with respect to width to make given h/w ratio
    h_rect = int(w_rect * height_over_width_ratio) 
    
    cropped_shape = shape - np.array([x_rect, y_rect])
    
    if y_rect + h_rect > img.shape[0]:
        img = cv.copyMakeBorder(img, 0, img.shape[0] + y_rect + h_rect, 0, 0, cv.BORDER_CONSTANT)
        
    return img[y_rect : y_rect + h_rect, x_rect : x_rect + w_rect], cropped_shape

    



def scale_img_and_shape(img, shape, height=320, width=240):
 
    img_height, img_width, _ = img.shape
    
    fx = 1.0*width / img_width
    fy = 1.0*height / img_height
    
    scaled_img = cv.resize(img, (width, height))
    scaled_shape = shape * np.array([fx, fy])
    
    return scaled_img, scaled_shape



def crop_and_scale_img_and_shape(img, shape, height=320, width=240):
    
    cropped_img, cropped_shape = crop_face_and_shape(img, shape, 1.0*height / width)
    
    if len(cropped_img) == 0:
        return np.empty(0), np.empty(0)
    
    scaled_img, scaled_shape = scale_img_and_shape(cropped_img, cropped_shape, height, width)
    
    return scaled_img, scaled_shape



def reduce_images_and_shapes(images, shapes, mode):
    
    reduced_images_dir_name = 'reduced_images_' + mode + '/'  
    reduced_shapes_dir_name = 'reduced_shapes_' + mode + '/'

    call(['mkdir', reduced_images_dir_name, reduced_shapes_dir_name])

    detection_fails = 0

    for i in range(len(images)):
        reduced_img, reduced_shape = crop_and_scale_img_and_shape(images[i], shapes[i])
    
        if len(reduced_img) == 0:
            detection_fails += 1            
            continue
        
        else:
            cv.imwrite(reduced_images_dir_name + '/' + str(i) +'.png', reduced_img)
            np.save(reduced_shapes_dir_name + '/' + str(i), reduced_shape)

    return detection_fails


def main():
    
    if(len(sys.argv) != 3):
        
        print('usage: ' + sys.argv[0] + ' data_dir_path' + ' mode')
        print('(mode can be train or test)')
        return


    data_dir_name = sys.argv[1]
    mode = sys.argv[2]

    print('-- Data reducing...')

    images, shapes = read_data(data_dir_name)

    detection_fails = reduce_images_and_shapes(images, shapes, mode)
    print('-- Data reduced. Face detector could not detect faces on ' + str(detection_fails) + ' images.')


main()
