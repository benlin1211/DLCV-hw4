_base_ = './DirectVoxGO/configs/default.py'

expname = 'dlcv_lego'
basedir = './p1_checkpoints_blendedmvs/'

data = dict(
    datadir='./hw4_data/hotdog/',
    dataset_type='blendedmvs',
    white_bkgd=True,
)
