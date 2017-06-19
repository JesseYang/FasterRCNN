import numpy as np

from cfgs.config import cfg


from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms

def bbox_transform(ex_rois, gt_rois):
  ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
  ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
  ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
  ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

  gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
  gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
  gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
  gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

  targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
  targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
  targets_dw = np.log(gt_widths / ex_widths)
  targets_dh = np.log(gt_heights / ex_heights)

  targets = np.vstack(
    (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
  return targets


def bbox_transform_inv(boxes, deltas):
  if boxes.shape[0] == 0:
    return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

  boxes = boxes.astype(deltas.dtype, copy=False)
  widths = boxes[:, 2] - boxes[:, 0] + 1.0
  heights = boxes[:, 3] - boxes[:, 1] + 1.0
  ctr_x = boxes[:, 0] + 0.5 * widths
  ctr_y = boxes[:, 1] + 0.5 * heights

  dx = deltas[:, 0::4]
  dy = deltas[:, 1::4]
  dw = deltas[:, 2::4]
  dh = deltas[:, 3::4]
  
  pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
  pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
  pred_w = np.exp(dw) * widths[:, np.newaxis]
  pred_h = np.exp(dh) * heights[:, np.newaxis]

  pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
  # x1
  pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
  # y1
  pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
  # x2
  pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
  # y2
  pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

  return pred_boxes


def clip_boxes(boxes, img_shape):
  """
  Clip boxes to image boundaries.
  """

  # x1 >= 0
  boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], img_shape[1] - 1), 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], img_shape[0] - 1), 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], img_shape[1] - 1), 0)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], img_shape[0] - 1), 0)
  return boxes



def proposal_layer(rpn_cls_prob, rpn_bbox_pred, img_shape):
    """A simplified version compared to fast/er RCNN
         For details please see the technical report
    """

    # Get the scores and bounding boxes
    scores = rpn_cls_prob[:, :, :, cfg.anchor_num:]
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    scores = scores.reshape((-1, 1))
    proposals = bbox_transform_inv(cfg.anchors, rpn_bbox_pred)
    proposals = clip_boxes(proposals, img_shape)

    # Pick the top region proposals
    order = scores.ravel().argsort()[::-1]
    if cfg.pre_nms_topN > 0:
        order = order[:cfg.pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # Non-maximal suppression
    keep = nms(np.hstack((proposals, scores)), cfg.nms_thresh)

    # Pick th top region proposals after NMS
    if cfg.post_nms_topN > 0:
        keep = keep[:cfg.post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return blob, scores
