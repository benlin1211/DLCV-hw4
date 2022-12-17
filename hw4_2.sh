#!/bin/bash
python3 eval4_2_downstream.py --csv_path=$1 --data_path=$2 --output_name=$3

# bash hw4_2.sh hw4_data/office/test.csv hw4_data/office/test/ hw4/output_p2/test_pred.csv

# $1: path to the images csv file (e.g., hw4_data/office/test.csv)
# $2: path to the folder containing images (e.g. hw4_data/office/test/)
# $3: path of output csv file (predicted labels) (e.g., hw4/output_p2/test_pred.csv)