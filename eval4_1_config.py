_base_ = './DirectVoxGO/configs/default.py'

expname = 'dlcv_lego'
basedir = './logs_4_1/'

data = dict(
    # datadir='./hw4_data/hotdog/',
    dataset_type='blender',
    white_bkgd=True,
)