#!/bin/bash
# TODO - run your inference Python3 code

# bash train4_1_report.sh './hw4_data/hotdog/transforms_val.json' './output_p1/'
# bash train4_1_report.sh './hw4_data_test/hotdog/transforms_test.json' './output_p1/'
#
# # $1: path to the transform_test.json (e.g., */*/transform_test.json)
# # which contains camera poses with the same format as in transfor_train.json, you should predict novel views base on this file.
# # $2: path of the folder to put output image (e.g., *.png)
# # the filename should be same as stated in transform_test.json file. The image size should be the same as training set, 800x800 pixel.

python DirectVoxGO/run_eval.py --config ./train4_1_config_report.py --json_dir $1 --output_dir $2  --render_val --render_only 
# python DirectVoxGO/run_eval.py --config ./train4_1_config_report2.py --json_dir $1 --output_dir $2  --render_val --render_only 


# python grade.py './output_p1/' './hw4_data/hotdog/val' 
# # $1 is the path to the folder of generated image (white background)
# # $2 is the path to the folder of gt image

