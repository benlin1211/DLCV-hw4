#!/bin/bash

# A
python train4_2_downstream.py --load_pretrain=False --pretrain_path='' --ckpt_path='ckpt4_2A'

# B
python train4_2_downstream.py --pretrain_path='./hw4_data/pretrain_model_SL.pt' --ckpt_path='ckpt4_2B'

# C 
python train4_2_downstream.py --pretrain_path='./ckpt4_2_naive_backbone/improved-net.pt' --ckpt_path='ckpt4_2C'

# D
python train4_2_downstream.py --pretrain_path='./hw4_data/pretrain_model_SL.pt' --ckpt_path='ckpt4_2D' --freeze_backbone=True

# E
python train4_2_downstream.py --pretrain_path='./ckpt4_2_naive_backbone/improved-net.pt' --ckpt_path='ckpt4_2E' --freeze_backbone=True
