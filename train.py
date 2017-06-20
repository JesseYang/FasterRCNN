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

from utils import proposal_layer

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
        return [InputDesc(tf.uint8, [1, None, None, 3], 'image'),
                InputDesc(tf.int32, [2], 'img_shape'),
                InputDesc(tf.int32, [None, 4], 'gt_boxes'),
                InputDesc(tf.int32, [None], 'gt_classes'),
                InputDesc(tf.int32, [1, None, None, cfg.anchor_num], 'rpn_label'),
                InputDesc(tf.int32, [1, None, None, cfg.anchor_num, 4], 'rpn_bbox_targets')]

    def _proposal_layer(self, rpn_class_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            rois, rpn_scores = tf.py_func(proposal_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self.img_shape],
                                          [tf.float32, tf.float32])
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = cfg.bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + \
                      (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = cfg.bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    def _build_graph(self, inputs):
        image, height, width, gt_boxes, gt_classes, rpn_label, rpn_bbox_targets = inputs
        image = tf.cast(image, tf.float32) * (1.0 / 255)

        # Wrong mean/std are used for compatibility with pre-trained models.
        # Should actually add a RGB-BGR conversion here.
        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - image_mean) / image_std
        image = tf.transpose(image, [0, 3, 1, 2])

        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                return Conv2D('convshortcut', l, n_out, 1, stride=stride)
            else:
                return l

        def basicblock(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
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
            ch_in = l.get_shape().as_list()[1]
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
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'):
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
                               channel=cfg.anchor_num * 2,
                               kernel_shape=1,
                               'VALID')

        rpn_cls_prob = tf.nn.softmax(rpn_cls_score, name='rpn_cls_prob', dim=1)
        rpn_bbox_pred = Conv2D("rpn_bbox_pred",
                               rpn,
                               channel=cfg.anchor_num * 4,
                               kernel_shape=1,
                               'VALID')

        rpn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

        rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                            rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

        # rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")

        # if is_training:
        #     rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        #     rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
        #     # Try to have a determinestic order for the computing graph, for reproducibility
        #     with tf.control_dependencies([rpn_labels]):
        #         rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
        # else:
        #     if cfg.TEST.MODE == 'nms':
        #         rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        #     elif cfg.TEST.MODE == 'top':
        #         rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        #     else:
        #         raise NotImplementedError


        # rcnn part





        prob = tf.nn.softmax(logits, name='output')
        prob = tf.identity(prob, name="NETWORK_OUTPUT")

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
        add_moving_summary(loss, wd_cost)
        self.cost = tf.add_n([loss, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test):
    isTrain = train_or_test == 'train'

    filename_list = cfg.train_list if isTrain else cfg.test_list
    ds = Data(filename_list)

    if isTrain:
        augmentors = [
            imgaug.RandomCrop(crop_shape=448),
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
            imgaug.Flip(horiz=True),
            imgaug.ToUint8()
        ]
    else:
        augmentors = [
            imgaug.RandomCrop(crop_shape=cfg.img_size),
            imgaug.ToUint8()
        ]
    ds = AugmentImageComponent(ds, augmentors)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    return ds


def get_config():
    dataset_train = get_data('train')
    dataset_val = get_data('test')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_val, [
                ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(0, 1e-2), (30, 3e-3), (60, 1e-3), (85, 1e-4), (95, 1e-5)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        model=Model(),
        max_epoch=110,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--batch_size', required=True)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    assert args.gpu is not None, "Need to specify a list of gpu for training!"
    NR_GPU = len(args.gpu.split(','))
    BATCH_SIZE = int(args.batch_size) // NR_GPU

    logger.auto_set_dir()
    config = get_config()
    if args.load:
        config.session_init = get_model_loader(args.load)
    config.nr_tower = NR_GPU
    SyncMultiGPUTrainer(config).train()
