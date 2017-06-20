import os, sys, shutil
import pickle
import numpy as np
import random
from scipy import misc
import six
from six.moves import urllib, range
import copy
import logging
import cv2
import json
import numpy.random as npr

from tensorpack import *

from cfgs.config import cfg
from utils import bbox_transform
from generate_anchors import generate_anchors_pre

import pdb

class Box():
    def __init__(self, p1, p2, p3, p4, mode='XYWH'):
        if mode == 'XYWH':
            # parameters: center_x, center_y, width, height
            self.x = p1
            self.y = p2
            self.w = p3
            self.h = p4
        if mode == "XYXY":
            # parameters: xmin, ymin, xmax, ymax
            self.x = (p1 + p3) / 2
            self.y = (p2 + p4) / 2
            self.w = p3 - p1
            self.h = p4 - p2

def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area

def box_union(box1, box2):
    i = box_intersection(box1, box2)
    u = box1.w * box1.h + box2.w * box2.h - i
    return u

def box_iou(box1, box2):
    return box_intersection(box1, box2) / box_union(box1, box2)

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K))
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

class Data(RNGDataFlow):
    def __init__(self, filename_list, shuffle, flip):
        self.filename_list = filename_list

        if isinstance(filename_list, list) == False:
            filename_list = [filename_list]

        content = []
        for filename in filename_list:
            with open(filename) as f:
                content.extend(f.readlines())

        self.imglist = [x.strip() for x in content] 
        self.shuffle = shuffle
        self.flip = flip
        # self.affine_trans = affine_trans

    def size(self):
        return len(self.imglist)

    def generate_sample(self, idx):
        line = self.imglist[idx]
        record = line.split(' ')
        record[1:] = [float(num) for num in record[1:]]

        image = cv2.imread(record[0])
        img_shape = image.shape
        img_size_min = np.min(img_shape[0:2])
        img_size_max = np.max(img_shape[0:2])
        img_scale = float(cfg.target_size) / float(img_size_min)

        # resize the image to make the shortest length 600 pixel
        image = cv2.resize(image, None, None, fx=img_scale, fy=img_scale,
                         interpolation=cv2.INTER_LINEAR)
        height, width, _ = image.shape
        feat_height, feat_width = np.ceil(np.asarray([height, width]) / cfg.feat_stride).astype(int)

        # generate gt boxes, including coordinates and classes
        i = 1
        gt_classes = []
        gt_boxes = []
        while i < len(record):
            # make coordinates of gt boxes from 1-based to 0-based
            # all gt boxes should be scaled
            xmin, ymin, xmax, ymax = [(ele - 1) * img_scale for ele in record[i:i+4]]
            class_num = int(record[i + 4])

            gt_boxes.append([xmin, ymin, xmax, ymax])
            gt_classes.append(class_num)

            i += 5

        gt_boxes = np.asarray(gt_boxes)
        gt_classes = np.asarray(gt_classes)

        # generate anchors, only keep anchors inside the image
        all_anchors, _ = generate_anchors_pre(feat_height, feat_width, cfg.feat_stride, anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2))

        total_anchors = all_anchors.shape[0]

        _allowed_border = 0
        inds_inside = np.where(
            (all_anchors[:, 0] >= -_allowed_border) &
            (all_anchors[:, 1] >= -_allowed_border) &
            (all_anchors[:, 2] < width + _allowed_border) &
            (all_anchors[:, 3] < height + _allowed_border)
        )[0]

        anchors = all_anchors[inds_inside, :]

        # calculate ious between anchors and gt boxes
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))

        # for each anchor, get its max iou with some gt box
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

        # for each gt box, find the anchor which has maximum iou with it
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)

        print(gt_argmax_overlaps.shape)
        print(np.where(max_overlaps >= cfg.rpn_positive_iou_th)[0].shape)

        # assign positive and negative ious
        labels[max_overlaps < cfg.rpn_negative_iou_th] = 0
        labels[gt_argmax_overlaps] = 1
        labels[max_overlaps >= cfg.rpn_positive_iou_th] = 1

        # subsample positive labels if we have too many
        num_fg = int(cfg.rpn_fg_fraction * cfg.rpn_batch)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.rpn_batch - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1


        # Compute bounding-box regression targets for an image.
        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = bbox_transform(anchors, gt_boxes[argmax_overlaps, :])

        # only the positive ones have regression targets
        # bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        # bbox_inside_weights[labels == 1, :] = np.array(cfg.rpn_bbox_inside_weights)

        # bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        # num_examples = np.sum(labels >= 0)
        # positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        # negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        # bbox_outside_weights[labels == 1, :] = positive_weights
        # bbox_outside_weights[labels == 0, :] = negative_weights

        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        # bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        # bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        # labels
        labels = labels.reshape((1, feat_height, feat_width, cfg.anchor_num))
        # labels = labels.reshape((1, feat_height, feat_width, cfg.anchor_num)).transpose(0, 3, 1, 2)
        # labels = labels.reshape((1, 1, cfg.anchor_num * feat_height, feat_width))
        rpn_labels = labels

        # bbox_targets
        bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, cfg.anchor_num, 4))
        rpn_bbox_targets = bbox_targets

        # bbox_inside_weights
        # bbox_inside_weights = bbox_inside_weights.reshape((1, feat_height, feat_width, cfg.anchor_num * 4))
        # rpn_bbox_inside_weights = bbox_inside_weights

        # bbox_outside_weights
        # bbox_outside_weights = bbox_outside_weights.reshape((1, feat_height, feat_width, cfg.anchor_num * 4))
        # rpn_bbox_outside_weights = bbox_outside_weights

        # return [image, [height, width], gt_boxes, gt_classes, rpn_labels, rpn_bbox_targets, all_anchors]
        image = np.expand_dims(image, 0)
        img_shape = np.asarray([height, width])
        print(image.shape)
        print(img_shape.shape)
        print(gt_boxes.shape)
        print(gt_classes.shape)
        print(rpn_labels.shape)
        print(rpn_bbox_targets.shape)
        return [image, img_shape, gt_boxes, gt_classes, rpn_labels, rpn_bbox_targets]

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield self.generate_sample(k)

if __name__ == '__main__':
    pass
