#!/usr/bin/env python

import cv2 as cv
import sys
from subprocess import call
import numpy as np
from keras.models import load_model
from patches_creation import create_patches, create_dict_input
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
from prepare_uploaded_image import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt


initial_model = load_model('initial_shape_model.hdf5') 
self_updating_cnn_model = load_model('self_updating_CNN.hdf5')

class MyWindow(Gtk.ApplicationWindow):
    # create a window

    def __init__(self, app):
        Gtk.Window.__init__(self, title="Catoonizer", application=app)
        self.set_default_size(900, 800)

        vbox = Gtk.VBox()
        
        cover_box = Gtk.VBox()
        button_box = Gtk.VBox()
        images_box = Gtk.HBox()
    

        button_upload_file = Gtk.Button("Upload image file")

        button_upload_file.connect("clicked", self.upload_file)

        button_box.pack_end(button_upload_file, 0, 0, 0)
           
        self.cover = Gtk.Image()
        self.image1 = Gtk.Image()
        self.image2 = Gtk.Image()
        self.image3 = Gtk.Image()        
    
        self.label1 = Gtk.Label()
        self.label2 = Gtk.Label()
        self.label3 = Gtk.Label()

        self.cover.set_from_file("resources/cat_halloween.jpg")
        self.image1.set_from_file("resources/question.png")
        self.image2.set_from_file("resources/question.png")
        self.image3.set_from_file("resources/question.png")

        label_image_box1 = Gtk.VBox()
        label_image_box2 = Gtk.VBox()
        label_image_box3 = Gtk.VBox()

        label_image_box1.add(self.label1)
        label_image_box2.add(self.label2)
        label_image_box3.add(self.label3)
        
        label_image_box1.add(self.image1)
        label_image_box2.add(self.image2)
        label_image_box3.add(self.image3)

        cover_box.pack_start(self.cover, 0, 0, 0)

        images_box.add(label_image_box1)
        images_box.add(label_image_box2)
        images_box.add(label_image_box3)

        vbox.add(cover_box)
        vbox.add(images_box)
        vbox.add(button_box)


        self.add(vbox)

        
    def upload_file(self, widget):
        dialog = Gtk.FileChooserDialog("Upload frontal face image...", self,
            Gtk.FileChooserAction.OPEN,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
             Gtk.STOCK_OPEN, Gtk.ResponseType.ACCEPT))

        self.add_filters(dialog)

        response = dialog.run()
        if response == Gtk.ResponseType.ACCEPT:

            img_path = dialog.get_filename()
            print(img_path)

            img = cv.imread(img_path)

            self.predict_on_single_image(img)

        dialog.destroy()
        



    def add_filters(self, dialog):
        
        filter_jpg = Gtk.FileFilter()
        filter_jpg.set_name("jpg files")
        filter_jpg.add_pattern("*.jpg")

        filter_png = Gtk.FileFilter()
        filter_png.set_name("png files")
        filter_png.add_pattern("*.png")

        dialog.add_filter(filter_jpg)
        dialog.add_filter(filter_png)



    def predict_on_single_image(self, img):

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

        call(['./exec.sh'])

        show_result('face_image.png', y_current_shape_prediction)

        self.image1.set_from_file("face_image.png")
        self.image2.set_from_file("detected_points.png")
        self.image3.set_from_file("head_with_body.png")
            
        self.label1.set_markup('Face image:')
        self.label2.set_markup('Detected key facial landmarks:')
        self.label3.set_markup('Catoonized:')


        
def show_result(img_path, shape):
    img = cv.imread(img_path)    

    plt.axis('off')

    plt.imshow(img)
    plt.scatter(shape.reshape(68, 2)[:, 0], shape.reshape(68, 2)[:, 1], color='yellow', s=7)  
    plt.savefig('detected_points.png', bbox_inches='tight')
    plt.close('all')



class MyApplication(Gtk.Application):

    def __init__(self):
        Gtk.Application.__init__(self)

    def do_activate(self):
        win = MyWindow(self)
        win.show_all()

    def do_startup(self):
        Gtk.Application.do_startup(self)

app = MyApplication()
exit_status = app.run(sys.argv)
sys.exit(exit_status)
