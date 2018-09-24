import numpy as np


# for given prediction of initial model, prepare image patches with centers in predicted points 
def prepare_patches_data(x, y_predicted):
    
    all_images_patches = []

    flatten_shape_length = y_predicted.shape[1]

    for i in range(len(x)):
        patches = create_patches(x[i], y_predicted[i].reshape(flatten_shape_length / 2, 2))
        all_images_patches.append(patches)
    
    result = np.array(all_images_patches)

    self_updating_cnn_input = create_dict_input(result)

    return self_updating_cnn_input


# create single square patch of image which center is (x, y), and of (d x d) dimension
def create_patch(img, point, d=22):
 
    x, y = point.astype(int)
    img_h, img_w, _ = img.shape
    
    x = max(d + 1, x)
    y = max(d + 1, y)
    
    x = min(x, img_w - d - 1)
    y = min(y, img_h - d - 1)
  
    patch = img[y - d : y + d + 1, x - d : x + d + 1]
    
    return patch



def create_patches(img, shape, d=22):
    
    padded_img = np.pad(img, ((d, d), (d, d), (0, 0)), 'constant')
    
    translated_shape = shape + np.array([d, d])
    all_patches = []
    
    for point in translated_shape:
        single_patch = create_patch(padded_img, point, d)
        all_patches.append(single_patch)
            
    return np.array(all_patches)



def create_dict_input(x):    

    inputs = {}    
    for i in range(68):
        inputs['input_' + str(i)] = x[:, i]
    
    return inputs


