#!/bin/bash
python eval4_2_downstream.py --resume_path="./ckpt4_2A/downstring_SGD.pth"
python grade_p2.py 
python eval4_2_downstream.py --resume_path="./ckpt4_2B/downstring_SGD.pth"
python grade_p2.py 
python eval4_2_downstream.py --resume_path="./ckpt4_2C/downstring_SGD.pth"
python grade_p2.py 
python eval4_2_downstream.py --resume_path="./ckpt4_2D/downstring_SGD.pth"
python grade_p2.py 
python eval4_2_downstream.py --resume_path="./ckpt4_2E/downstring_SGD.pth"
python grade_p2.py 
