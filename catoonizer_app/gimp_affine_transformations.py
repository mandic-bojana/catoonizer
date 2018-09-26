import math
from gimp_coordinates import *
from gimpfu import *


def rotate_to_align(image, item, shape):
    
    mapping = map_coordinates(shape)    

    x1, y1 = mapping['beard_left_1']
    x2, y2 = mapping['beard_right_8']

    theta = math.atan2((y2 - y1), (x2 - x1))
    
    print(theta)

    item = pdb.gimp_item_transform_rotate(item, -theta, True, 0, 0)

    img_center_x = image.width / 2
    img_center_y = image.height / 2

    
    new_shape = [[(x - img_center_x) * math.cos(theta) + (y - img_center_y) * math.sin(theta) + img_center_x, 
                  -(x - img_center_x) * math.sin(theta) + (y - img_center_y) * math.cos(theta) + img_center_y] for [x, y] in shape]
   

    return item, new_shape



def translate_shape_to_selection(shape, x_top_left, y_top_left):

    new_shape =  [[x - x_top_left, y - y_top_left] for [x, y] in shape]
    return new_shape


def scale_shape_to_dimensions(image, shape, new_width, new_height):
    
    old_width = image.width
    old_height = image.height

    x_scale_factor = 1.0 * new_width / old_width
    y_scale_factor = 1.0 * new_height / old_height

    new_shape =  [[x * x_scale_factor, y * y_scale_factor] for [x, y] in shape]
    return new_shape


