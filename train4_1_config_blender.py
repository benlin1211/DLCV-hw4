_base_ = './DirectVoxGO/configs/default.py'

expname = 'dlcv_lego'
basedir = './p1_checkpoints_blender/'

data = dict(
    datadir='./hw4_data/hotdog/',
    dataset_type='blender',
    white_bkgd=True,
)
