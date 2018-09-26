#!/bin/bash

gimp -idf --batch-interpreter=python-fu-eval -b 'import sys; sys.path=["."]+sys.path;import gimp_script;gimp_script.create_cat_image("face_image.png", "coordinates.txt")'

