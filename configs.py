import argparse
from yacs.config import CfgNode as CN

cfg = CN()


def init_config():
    cfg.detector = CN()
    cfg.detector.method = 'skeleton'
    cfg.detector.cut_hor = 770
    cfg.detector.cut_vet = 240
    cfg.detector.cut_bot = 280
    # cfg.detector.pad = 32
    cfg.detector.cont_len_thres = 30
    cfg.detector.img_size_x = 4024
    cfg.detector.img_size_y = 3036

    # cfg.parallel = CN()

    cfg.skeleton = CN()
    cfg.skeleton.max_res = 100
    cfg.skeleton.method = 'twice'

    cfg.hybrid = CN()
    cfg.hybrid.max_res = 600
    cfg.hybrid.method = 'twice'
    cfg.hybrid.patch_size = 50
    cfg.hybrid.dist_thres_high = 50.0
    cfg.hybrid.dist_thres_low = 20.0
    cfg.hybrid.patch_pad = 30

    return cfg


def get_config():
    return cfg
