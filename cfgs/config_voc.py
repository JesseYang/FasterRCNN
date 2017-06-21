import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.target_size = 600
cfg.feat_stride = 16

cfg.rpn_positive_iou_th = 0.7
cfg.rpn_negative_iou_th = 0.3
cfg.rpn_batch = 256
cfg.rpn_fg_fraction = 0.5

cfg.bbox_outside_weights = 
cfg.bbox_inside_weights = 

cfg.anchor_scales = (8, 16, 32)
cfg.anchor_ratios = (0.5, 1, 2)
cfg.anchor_num = 9

cfg.rpn_reg_weight = 1 / cfg.rpn_batch

cfg.pre_nms_topN = 12000
cfg.post_nms_topN = 2000

cfg.rpn_nms_th = 0.7

cfg.rcnn_batch_size = 128
cfg.rcnn_fg_frac = 0.25

cfg.fg_iou_th = 0.5

cfg.bg_iou_hi = 0.5
cfg.bg_iou_lo = 0.1

cfg.n_classes = 20

cfg.threshold = 0.6

cfg.weight_decay = 5e-4

# ignore boxes which are too small (height or width smaller than size_th * 32)
cfg.size_th = 0.1

cfg.classes_name =  ["aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat",
                     "chair", "cow", "diningtable", "dog",
                     "horse", "motorbike", "person", "pottedplant",
                     "sheep", "sofa", "train","tvmonitor"]

cfg.classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
	               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
	               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
	               'sofa': 17, 'train': 18, 'tvmonitor': 19}


cfg.train_list = ["voc_2007_train.txt", "voc_2012_train.txt", "voc_2007_val.txt", "voc_2012_val.txt"]
cfg.test_list = "voc_2007_test.txt"

cfg.det_th = 0.001
cfg.iou_th = 0.5
cfg.nms = True
cfg.nms_th = 0.45

cfg.mAP = True
