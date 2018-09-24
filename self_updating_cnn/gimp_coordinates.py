import re

body_image_dims = (756, 916)
ears_image_dims = (258, 399)
    
ears_left_connect_point = [306, 182]
ears_right_connect_point = [497, 175]


def head_bounding_box_for_body():
    
    height = 320
    width = 240

    top_left_corner_x = 270    
    top_left_corner_y = 30        

    return top_left_corner_x, top_left_corner_y, height, width


def head_bounding_box_for_ears():
    height = 169
    width = 200    

    top_left_corner_x = 33    
    top_left_corner_y = 150

    height = 169
    width = 200

    return top_left_corner_x, top_left_corner_y, height, width

    

def map_coordinates(shape):

    mapping = {}

    #  BEARD -------------------------
    
    mapping['beard_left_1'] = shape[0]
    mapping['beard_left_2'] = shape[1]
    mapping['beard_left_3'] = shape[2]
    mapping['beard_left_4'] = shape[3]
    mapping['beard_left_5'] = shape[4]
    mapping['beard_left_6'] = shape[5]
    mapping['beard_left_7'] = shape[6]
    mapping['beard_left_8'] = shape[7]
    
    mapping['beard_middle'] = shape[8]

    mapping['beard_right_1'] = shape[9]
    mapping['beard_right_2'] = shape[10]
    mapping['beard_right_3'] = shape[11]
    mapping['beard_right_4'] = shape[12]
    mapping['beard_right_5'] = shape[13]
    mapping['beard_right_6'] = shape[14]
    mapping['beard_right_7'] = shape[15]
    mapping['beard_right_8'] = shape[16]

    #  EYEBROWS ----------------------

    mapping['eyebrow_left_1'] = shape[17]
    mapping['eyebrow_left_2'] = shape[18]
    mapping['eyebrow_left_3'] = shape[19]
    mapping['eyebrow_left_4'] = shape[20]
    mapping['eyebrow_left_5'] = shape[21]

    mapping['eyebrow_right_1'] = shape[22]
    mapping['eyebrow_right_2'] = shape[23]
    mapping['eyebrow_right_3'] = shape[24]
    mapping['eyebrow_right_4'] = shape[25]
    mapping['eyebrow_right_5'] = shape[26]


    # NOSE

    mapping['nose_vertical_1'] = shape[27]
    mapping['nose_vertical_2'] = shape[28]
    mapping['nose_vertical_3'] = shape[29]
    mapping['nose_vertical_4'] = shape[30]

    mapping['nose_horizontal_1'] = shape[31]
    mapping['nose_horizontal_2'] = shape[32]
    mapping['nose_horizontal_3'] = shape[33]
    mapping['nose_horizontal_4'] = shape[34]
    mapping['nose_horizontal_5'] = shape[35]

    # EYES ---------------------------------

    mapping['eye_left_outer_corner'] = shape[36]
    mapping['eye_left_upper_1'] = shape[37]
    mapping['eye_left_upper_2'] = shape[38]
    mapping['eye_left_inner_corner'] = shape[39]
    mapping['eye_left_lower_1'] = shape[40]
    mapping['eye_left_lower_2'] = shape[41]

    
    mapping['eye_right_inner_corner'] = shape[42]
    mapping['eye_right_upper_1'] = shape[43]
    mapping['eye_right_upper_2'] = shape[44]
    mapping['eye_right_outer_corner'] = shape[45]
    mapping['eye_right_lower_1'] = shape[46]
    mapping['eye_right_lower_2'] = shape[47]


    # MOUTH -----------------------------

    mapping['mouth_left_corner'] = shape[48]
    mapping['mouth_left_upper_1'] = shape[49]
    mapping['mouth_left_upper_2'] = shape[50]

    mapping['mouth_middle_upper'] = shape[51]

    mapping['mouth_left_upper_1'] = shape[52]
    mapping['mouth_left_upper_2'] = shape[53]
    mapping['mouth_right_corner'] = shape[54]

    mapping['mouth_right_lower_1'] = shape[55]
    mapping['mouth_right_lower_2'] = shape[56]

    mapping['mouth_midle_lower'] = shape[57]

    mapping['mouth_left_lower_1'] = shape[58]
    mapping['mouth_left_lower_2'] = shape[59]

    return mapping




def face_bounding_path(shape):

    beard_path = shape[:17]
    eyebrows_path = shape[17:27]
    eyebrows_path.reverse()

    result = beard_path + eyebrows_path    

    return result
    

def left_eye_bounding_path(shape):
    
    return shape[36:42]


def right_eye_bounding_path(shape):
        
    return shape[42:48]  

def left_eyelid_path(shape):
    
    return shape[36:40]

def right_eyelid_path(shape):
    
    return shape[42:46]


def mouth_bounding_path(shape):
        
    return shape[48:60]  


def nose_circle_path(shape):
    return shape[30:36]



def read_shape_coordinates(coordinates_file):

    point_regex = re.compile('(-?\d+.\d*) (-?\d+.\d*)')
    annot_file = open(coordinates_file)
    file_lines = annot_file.readlines()
        
    shape = []

    for line in file_lines:
        x_coord = float(re.match(point_regex, line).group(1))
        y_coord = float(re.match(point_regex, line).group(2))
        point = [x_coord, y_coord]
        shape.append(point)

    return shape




