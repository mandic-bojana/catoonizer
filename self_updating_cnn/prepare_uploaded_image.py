import cv2 as cv
import numpy as np

def write_coordinates(shape):
    
    coordinates_file = open('coordinates.txt', 'w')

    for x, y in shape.reshape(68, 2):
        coordinates_file.write(str(x) + ' ' + str(y) + '\n')


def detect_crop_and_rescale_face(img, height_over_width_ratio=1.0*320 / 240):


    face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_detector.load('../utils/haarcascade_frontalface_default.xml')
    detection = face_detector.detectMultiScale(img, minNeighbors=10, scaleFactor=1.1)
    
    if len(detection) < 1:
        return np.empty(0)
    
    max_detected_rec = detection[0]

    for i in range(len(detection)):
        if rect_area(detection[i]) > rect_area(max_detected_rec):
            max_detected_rec = detection[i]
    
    x_rect, y_rect, w_rect, h_rect = max_detected_rec


    border = np.random.randint(0, 30)
    
    x_rect += border
    y_rect += border
    w_rect += border
    h_rect += border
    

    if y_rect < 0:
        img = cv.copyMakeBorder(img, y_rect, 0, 0, 0, cv.BORDER_CONSTANT)
    if y_rect + h_rect > img.shape[0]:
        img = cv.copyMakeBorder(img, 0, y_rect + h_rect - img.shape[0], 0, 0, cv.BORDER_CONSTANT)
    if x_rect < 0:
        img = cv.copyMakeBorder(img, 0, 0, x_rect, 0, cv.BORDER_CONSTANT)
    if x_rect + w_rect > img.shape[1]:
        img = cv.copyMakeBorder(img, 0, 0, 0, x_rect + w_rect - img.shape[1], cv.BORDER_CONSTANT)
    

    h_rect = int(w_rect * height_over_width_ratio) 
    
    
    if y_rect + h_rect > img.shape[0]:
        img = cv.copyMakeBorder(img, 0, img.shape[0] + y_rect + h_rect, 0, 0, cv.BORDER_CONSTANT)
        
    cropped_img = img[y_rect : y_rect + h_rect, x_rect : x_rect + w_rect]
    scaled_img = scale_image(cropped_img)
   
    return scaled_img
    

def rect_area(rect_points):
    _, _, h, w = rect_points
    return h * w



def scale_image(img, height=320, width=240):
 
    img_height, img_width, _ = img.shape
    
    fx = 1.0*width / img_width
    fy = 1.0*height / img_height
    
    scaled_img = cv.resize(img, (width, height))
    
    return scaled_img


