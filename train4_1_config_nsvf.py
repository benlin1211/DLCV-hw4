_base_ = './DirectVoxGO/configs/default.py'

expname = 'dlcv_lego'
basedir = './p1_checkpoints_nsvf/'

data = dict(
    datadir='./hw4_data/hotdog/',
    dataset_type='nsvf',
    white_bkgd=True,
)
