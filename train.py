import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from reader import *
from cfgs.config import cfg

import pdb

from utils import proposal_layer, proposal_target_layer

class Model(ModelDesc):

    def __init__(self, mode):
        super(Model, self).__init__()
        self._mode = mode

    def _get_inputs(self):
        # inputs should include:
        # 1. img_shape
        # 2. all the anchors
        # 3. gt boxes
        # 4. target for rpn part, should be calculated from anchors and gt boxes
        # 5. pay attention that target for fast rcnn part cannot be provided,
        #    since they are calculated based on the rpn output
        #    (selecting positive and negative samples from roi, which is output of rpn)
        # 6. pay attention that the network only supports one image each time
        return [InputDesc(tf.uint8, [1, None, None, 3], 'image'),
                InputDesc(tf.int32, [2], 'img_shape'),
                InputDesc(tf.int32, [None, 4], 'gt_boxes'),
                InputDesc(tf.int32, [None], 'gt_classes'),
                InputDesc(tf.int32, [1, None, None, cfg.anchor_num], 'rpn_label'),
                InputDesc(tf.float32, [1, None, None, cfg.anchor_num, 4], 'rpn_bbox_targets'),
                InputDesc(tf.uint8, [None, 4], "anchors")]

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_label=None, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        box_diff = bbox_label * box_diff
        abs_box_diff = tf.abs(box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
        loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        loss_box = tf.reduce_mean(tf.reduce_sum(
            loss_box,
            axis=dim
        ))
        return loss_box

    def _build_graph(self, inputs):
        # image, height, width, gt_boxes, gt_classes, rpn_label, rpn_bbox_targets, anchors = inputs
        image, img_shape, gt_boxes, gt_classes, rpn_label, rpn_bbox_targets, anchors = inputs
        height = img_shape[0]
        width = img_shape[1]
        image = tf.cast(image, tf.float32) * (1.0 / 255)

        # Wrong mean/std are used for compatibility with pre-trained models.
        # Should actually add a RGB-BGR conversion here.
        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - image_mean) / image_std
        # image = tf.transpose(image, [0, 3, 1, 2])

        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                return Conv2D('convshortcut', l, n_out, 1, stride=stride)
            else:
                return l

        def basicblock(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[3]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3)
            return l + shortcut(input, ch_in, ch_out, stride)

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[3]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1)
            return l + shortcut(input, ch_in, ch_out * 4, stride)

        def layer(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                return l

        net_cfg = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck)
        }
        defs, block_func = net_cfg[cfg.depth]

        # share part
        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NHWC'):
            features = (LinearWrap(image)
                        .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                        .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                        .apply(layer, 'group0', block_func, 64, defs[0], 1, first=True)
                        .apply(layer, 'group1', block_func, 128, defs[1], 2)
                        .apply(layer, 'group2', block_func, 256, defs[2], 2)())

        # rpn part
        rpn = (LinearWrap(features)
               .Conv2D('rpn_conv/3x3', 512, 3, stride=1)
               .BatchNorm('rpn_bn')
               .LeakyReLU('rpn_relu', 0.1)())

        rpn_cls_score = Conv2D('rpn_cls_score',
                               rpn,
                               out_channel=cfg.anchor_num * 2,
                               kernel_shape=1,
                               padding='VALID')
        # from 1 x feat_height x feat_width x (2 x anchor_num) to (feat_height x feat_width x anchor_num) x 2
        rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])

        rpn_cls_prob = tf.nn.softmax(rpn_cls_score, name='rpn_cls_prob', dim=1)
        rpn_bbox_pred = Conv2D("rpn_bbox_pred",
                               rpn,
                               out_channel=cfg.anchor_num * 4,
                               kernel_shape=1,
                               padding='VALID')
        temp_shape = tf.shape(rpn_bbox_pred)
        feat_height = temp_shape[1]
        feat_width = temp_shape[2]
        # from 1 x feat_height x feat_width x (4 x anchor_num) to 1 x feat_height x feat_width x anchor_num x 4
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [1, feat_height, feat_width, cfg.anchor_num, 4])

        rpn_cls_label = tf.reshape(rpn_label, [-1])

        rpn_select = tf.where(tf.not_equal(rpn_cls_label, -1))

        # only select the pred boxes that corresponds to positive and nagative anchors
        # rpn_cls_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
        rpn_cls_label = tf.gather_nd(rpn_cls_label, rpn_select)
        # rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
        rpn_cls_score = tf.gather_nd(rpn_cls_score, rpn_select)

        # the class loss for rpn
        rpn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_cls_label))

        # the box loss for rpn
        rpn_bbox_label = tf.tile(rpn_label, [1, 1, 1, 4])
        rpn_bbox_label = tf.reshape(rpn_bbox_label, (1, feat_height, feat_width, cfg.anchor_num, 4))
        rpn_bbox_label = tf.cast(rpn_bbox_label == 1, tf.float32)
        rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, bbox_label=rpn_bbox_label, sigma=3.0, dim=[1, 2, 3, 4])
        rpn_loss_box = rpn_loss_box / cfg.rpn_reg_weight

        # get roi proposals
        with tf.variable_scope("rois") as scope:
            rois, roi_scores = tf.py_func(proposal_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, height, width, anchors],
                                          [tf.float32, tf.float32])
        rois.set_shape([None, 4])
        roi_scores.set_shape([None, 1])

        # sample from roi proposals to get minibatch and targets for fast rcnn part
        with tf.variable_scope("rpn_rois") as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, gt_boxes, gt_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

        rois.set_shape([cfg.rcnn_batch_size, 4])
        roi_scores.set_shape([cfg.rcnn_batch_size])
        labels.set_shape([cfg.rcnn_batch_size])
        bbox_targets.set_shape([cfg.rcnn_batch_size, cfg.n_classes * 4])
        bbox_inside_weights.set_shape([cfg.rcnn_batch_size, cfg.n_classes * 4])
        bbox_outside_weights.set_shape([cfg.rcnn_batch_size, cfg.n_classes * 4])

        labels = tf.to_int32(labels, name="to_int32")

        # instead of doing RoI pooling, do crop and resize
        with tf.variable_scope("crop_and_resize") as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            feature_shape = tf.shape(features)
            height = (tf.to_float(feature_shape[1]) - 1.) * np.float32(cfg.feat_stride)
            width = (tf.to_float(feature_shape[2]) - 1.) * np.float32(cfg.feat_stride)
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be backpropagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
            # features = tf.stop_gradient(features)
            crops = tf.image.crop_and_resize(features,
                                             bboxes,
                                             tf.to_int32(batch_ids),
                                             [cfg.pooling_size, cfg.pooling_size],
                                             name="crops")

        ##### debug #####
        # import pdb
        # pdb.set_trace()
        # crops = tf.stop_gradient(crops)
        temp = tf.reduce_mean(crops)
        self.cost = tf.reduce_mean(rpn_loss_box) + temp
        return
        #################

        # last conv block for resnet
        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NHWC'):
            features = (LinearWrap(crops)
                        .apply(layer, 'group3', block_func, 512, defs[3], 1)
                        .BNReLU('bnlast')
                        .GlobalAvgPooling('gap')())

        # outputs
        cls_score = FullyConnected("cls_score", features, cfg.n_classes, nl=tf.identity)
        bbox_pred = FullyConnected("bbox_pred", features, cfg.n_classes * 4, nl=tf.identity)

        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=tf.reshape(cls_score, [-1, cfg.n_classes]), labels=labels))

        bbox_label = tf.tile(labels, [4])
        bbox_label = tf.reshape(bbox_label, (-1, 4))
        bbox_label = tf.cast(bbox_label > 0, tf.float32)
        loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_label=bbox_inside_weights)

        loss = rpn_loss_box + rpn_cross_entropy + loss_box + cross_entropy

        wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
        add_moving_summary(loss, wd_cost)
        self.cost = tf.add_n([loss, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test):
    isTrain = train_or_test == 'train'

    filename_list = cfg.train_list if isTrain else cfg.test_list
    ds = Data(filename_list, shuffle=isTrain, flip=isTrain)

    if isTrain:
        augmentors = [
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False),
                 imgaug.Saturation(0.4),
                 # rgb-bgr conversion
                 imgaug.Lighting(0.1,
                                 eigval=[0.2175, 0.0188, 0.0045][::-1],
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Clip(),
            imgaug.ToUint8()
        ]
    else:
        augmentors = [
            imgaug.ToUint8()
        ]
    # ds = AugmentImageComponent(ds, augmentors)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
    # ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    return ds


def get_config():
    dataset_train = get_data('train')
    dataset_val = get_data('test')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',
                                      [(0, 1e-2), (30, 3e-3), (60, 1e-3), (85, 1e-4), (95, 1e-5)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=Model("train"),
        max_epoch=110,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    assert args.gpu is not None, "Need to specify a list of gpu for training!"
    # NR_GPU = len(args.gpu.split(','))
    # BATCH_SIZE = int(args.batch_size) // NR_GPU

    logger.auto_set_dir()
    config = get_config()
    if args.load:
        config.session_init = get_model_loader(args.load)
    config.nr_tower = 1
    SyncMultiGPUTrainer(config).train()
