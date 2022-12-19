#!/bin/bash
# TODO - run your inference Python3 code

python3 DirectVoxGO/run_eval.py --json_dir $1 --output_dir $2 --config ./train4_1_config_blender.py --render_val --render_only

# bash hw4_1.sh './hw4_json_only/hotdog/transforms_test.json' './output_p1/'
#
# # $1: path to the transform_test.json (e.g., */*/transform_test.json)
# # which contains camera poses with the same format as in transfor_train.json, you should predict novel views base on this file.
# # $2: path of the folder to put output image (e.g., *.png)
# # the filename should be same as stated in transform_test.json file. The image size should be the same as training set, 800x800 pixel.

# python grade.py './output_p1/' './hw4_data/hotdog/val' 
# # $1 is the path to the folder of generated image (white background)
# # $2 is the path to the folder of gt image

