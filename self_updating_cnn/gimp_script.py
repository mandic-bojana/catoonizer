#!/usr/bin/python
from gimpfu import *
import math
from coordinates import *
from affine_transformations import *

    
def create_cat_image(filename, points_file):
    
  shape = read_shape_coordinates(points_file)

  extract_eyes_and_mouth(filename, shape)
  updated_shape = extract_face(filename, 'gimp_resources/eyes.png', shape)

  updated_shape = add_ears('gimp_resources/ears.png', 'gimp_resources/face.png', updated_shape)

  add_body('gimp_resources/body_for_head.png', 'gimp_resources/head_with_ears.png', updated_shape)
  
  
  pdb.gimp_quit(1)


def select_polygon_path(bounding_path, image, enable_feather_edges, radius):
    
    pdb.gimp_context_set_feather(enable_feather_edges)
    pdb.gimp_context_set_feather_radius(radius, radius)
    
    segs = [coordinate for point in bounding_path for coordinate in point]   
    pdb.gimp_image_select_polygon(image, CHANNEL_OP_ADD, len(segs), segs)
	


def extract_face(face_image_path, eyes_image_path, shape):
    
    image = pdb.gimp_file_load(face_image_path, face_image_path)

    bounding_path = face_bounding_path(shape)
    select_polygon_path(bounding_path, image, False, 0)

    pdb.plug_in_dog(image, pdb.gimp_image_get_active_layer(image), 170, 1, True, True)
    pdb.gimp_hue_saturation(pdb.gimp_image_get_active_layer(image), 0, 0, 10, -100)
    pdb.plug_in_colortoalpha(image, pdb.gimp_image_get_active_layer(image), (255, 255, 255))
    
    eyes_layer = pdb.gimp_file_load_layer(image, eyes_image_path)
    pdb.gimp_image_insert_layer(image, eyes_layer, None , 0)

    pdb.gimp_edit_copy(eyes_layer)
    floating = pdb.gimp_edit_paste (image.layers[0], True)
    pdb.gimp_floating_sel_anchor(floating)

    pdb.gimp_selection_none(image)

    draw_nose(image, shape)

    pdb.gimp_selection_all(image)
    layer = pdb.gimp_image_merge_visible_layers(image, 1) 
    pdb.gimp_selection_none(image)

    select_polygon_path(bounding_path, image, True, 3)

    pdb.gimp_selection_invert(image)

    pdb.gimp_edit_fill(pdb.gimp_image_get_active_layer(image), 2)
    pdb.plug_in_colortoalpha(image, pdb.gimp_image_get_active_layer(image), (255, 255, 255))

    pdb.gimp_selection_none(image)
    
    pdb.gimp_context_set_interpolation(2)
    rotated, rotated_shape = rotate_to_align(image, pdb.gimp_image_get_active_layer(image), shape) 
   
    rotated_bounding_path = face_bounding_path(rotated_shape)
    select_polygon_path(rotated_bounding_path, image, False, 0)

    _, x_top_left, y_top_left, _, _ = pdb.gimp_selection_bounds(image)
    print(x_top_left, y_top_left)

    translated_shape = translate_shape_to_selection(rotated_shape, x_top_left, y_top_left)
    
    pdb.plug_in_autocrop(image, pdb.gimp_image_get_active_layer(image))

    new_width = 202
    new_height = int((1.0 * image.height/image.width) * new_width)	   

    scaled_shape = scale_shape_to_dimensions(image, translated_shape, new_width, new_height)
    pdb.gimp_image_scale(image, new_width, new_height)

    pdb.gimp_file_save(image, pdb.gimp_image_get_active_layer(image), 'face.png', 'face.png')
    pdb.gimp_image_delete(image)   

    return scaled_shape




def extract_eyes_and_mouth(image_path, shape):

    image = pdb.gimp_file_load(image_path, image_path)
    
    left_eye_path = left_eye_bounding_path(shape)
    right_eye_path = right_eye_bounding_path(shape)
    mouth_path = mouth_bounding_path(shape)
    
    select_polygon_path(left_eye_path, image, True, 3)
    select_polygon_path(right_eye_path, image, True, 3)

    pdb.plug_in_oilify(image, pdb.gimp_image_get_active_layer(image), 5, 1)
    pdb.plug_in_softglow(image, pdb.gimp_image_get_active_layer(image), 1, 1, 1)
    pdb.gimp_brightness_contrast(pdb.gimp_image_get_active_layer(image), 90, 90)

    select_polygon_path(mouth_path, image, True, 3)
    
    pdb.gimp_posterize(pdb.gimp_image_get_active_layer(image), 10)

    pdb.gimp_selection_invert(image)

    pdb.gimp_edit_fill(pdb.gimp_image_get_active_layer(image), 2)
    pdb.plug_in_colortoalpha(image, pdb.gimp_image_get_active_layer(image), (255, 255, 255))
    
    pdb.gimp_selection_none(image)

    draw_eye_line(image, shape)
    draw_mouth_to_nose_line(image, shape)
    
    pdb.gimp_file_save(image, image.layers[0], 'eyes.png', 'eyes.png')
    pdb.gimp_image_delete(image)
    


def draw_nose(image, shape):
    
    mapping = map_coordinates(shape)
    
    nose_circle = nose_circle_path(shape)

    select_polygon_path(nose_circle, image, True, 3)

    x1, y1 = mapping['nose_horizontal_3']
    x2, y2 = mapping['nose_vertical_3']

    nose_color = (255, 68, 137)
    pdb.gimp_context_set_foreground(nose_color)
    
    pdb.gimp_edit_blend(pdb.gimp_image_get_active_layer(image), FG_TRANSPARENT_MODE, NORMAL_MODE, GRADIENT_RADIAL, 50, 
                    0, REPEAT_NONE, False, False, 0, 0, True, x1, y1, x2, y2 + 2)

    


def draw_eye_line(image, shape):
    
    left_eyelid = left_eyelid_path(shape)
    right_eyelid = right_eyelid_path(shape)

    black = (0, 0, 0)

    stroke(image, right_eyelid, black, 1, 100)
    stroke(image, left_eyelid, black, 1, 100)
    

    


def draw_mouth_to_nose_line(image, shape):

    mapping = map_coordinates(shape)    

    mouth_to_nose_path = [mapping['mouth_middle_upper'], mapping['nose_horizontal_3']]
    
    red = (255, 0, 0)
	
    stroke(image, mouth_to_nose_path, red, 2, 10)



def add_ears(ears_image_file, face_image_file, shape):
    
    ears_image = pdb.gimp_file_load(ears_image_file, ears_image_file)
    
    face_layer = pdb.gimp_file_load_layer(ears_image, face_image_file)
    pdb.gimp_image_insert_layer(ears_image, face_layer, None , 0)

    x_top_left, y_top_left, _, _ = head_bounding_box_for_ears()
    pdb.gimp_layer_translate(face_layer, x_top_left, y_top_left)

    new_shape = translate_shape_to_selection(shape, -x_top_left, -y_top_left)

    pdb.gimp_edit_copy(face_layer)
    floating = pdb.gimp_edit_paste (ears_image.layers[0], True)
 
    pdb.gimp_floating_sel_anchor(floating)

    layer = pdb.gimp_image_merge_visible_layers(ears_image, 0)

    pdb.plug_in_autocrop(ears_image, pdb.gimp_image_get_active_layer(ears_image))

    pdb.gimp_file_save(ears_image, layer, 'head_with_ears.png', 'head_with_ears.png')
    pdb.gimp_image_delete(ears_image)
    
    return new_shape




def add_body(body_image_file, head_image_file, shape):
    
    body_image = pdb.gimp_file_load(body_image_file, body_image_file)
    
    head_layer = pdb.gimp_file_load_layer(body_image, head_image_file)
    pdb.gimp_image_insert_layer(body_image, head_layer, None , 0)

    x_top_left, y_top_left, h, w = head_bounding_box_for_body()
    pdb.gimp_layer_translate(head_layer, x_top_left, y_top_left)

    new_shape = translate_shape_to_selection(shape, -x_top_left, -y_top_left)

    pdb.gimp_edit_copy(head_layer)
    floating = pdb.gimp_edit_paste (body_image.layers[0], True)
    pdb.gimp_floating_sel_anchor(floating)


    layer = pdb.gimp_image_merge_visible_layers(body_image, 0)
    
    mapping = map_coordinates(new_shape)

    stroke(body_image, [[294, 480], mapping['beard_left_5']], (130, 130, 130), 0.5, 100)
    stroke(body_image, [[596, 496], mapping['beard_right_4']], (190, 190, 190), 2.3, 100)

    left_beard_1_x, left_beard_1_y = mapping['beard_left_1']
    right_beard_8_x, right_beard_8_y = mapping['beard_right_8']

    stroke(body_image, [mapping['eyebrow_left_5'], mapping['eyebrow_right_1']], (255, 255, 255), 2.7, 100)
    stroke(body_image, [mapping['eyebrow_left_1'], [left_beard_1_x + 4, left_beard_1_y + 4]], (255, 255, 255), 3, 100)
    stroke(body_image, [mapping['eyebrow_right_5'], [right_beard_8_x - 4, right_beard_8_y - 4]], (255, 255, 255), 3, 100)
    stroke(body_image, [mapping['beard_left_1'], ears_left_connect_point], (128, 128, 128), 0.5, 80)
    stroke(body_image, [mapping['beard_right_8'], ears_right_connect_point], (128, 128, 128), 0.5, 80)


    pdb.gimp_file_save(body_image, layer, 'head_with_body.png', 'head_with_body.png')
    pdb.gimp_image_delete(body_image)
    
    return new_shape



def stroke(image, path_points, color, size, opacity):
        

    segs = [coordinate for point in path_points for coordinate in point * 3]

   
    pdb.gimp_context_set_foreground(color)
    pdb.gimp_context_set_opacity(opacity)
    
    pdb.gimp_context_set_paint_method('gimp-ink')
    pdb.gimp_context_set_ink_size(size)
    pdb.gimp_context_set_ink_blob_type(2)
    pdb.gimp_context_set_ink_speed_sensitivity(0.5)


    openPath = pdb.gimp_vectors_new(image, 'path')
   

    pdb.gimp_image_insert_vectors(image, openPath, None, -1)

    s_id = pdb.gimp_vectors_stroke_new_from_points(openPath, 0, len(segs), segs, False)
    
    openPath.visible=True

    pdb.gimp_edit_stroke_vectors(image.layers[0], openPath)
    

