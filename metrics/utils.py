from typing import List, Dict
import torch
import numpy as np

from evaluation.structures import BBox, ResponseTrack


PRINT_FORMAT = "{:<30s} {:<15s}"


def segment_iou(
    target_segment: np.ndarray, candidate_segments: np.ndarray
) -> np.ndarray:
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1 + 1).clip(0)
    # Segment union.
    segments_union = (
        (candidate_segments[:, 1] - candidate_segments[:, 0] + 1)
        + (target_segment[1] - target_segment[0] + 1)
        - segments_intersection
    )
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def interpolated_prec_rec(prec: np.ndarray, rec: np.ndarray) -> np.ndarray:
    """Interpolated AP - VOCdevkit from VOC 2011."""
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def spatial_iou(box1: BBox, box2: BBox) -> float:
    """
    Computes iou between two bounding boxes
    """
    xi_s = max(box1.x1, box2.x1)
    xi_e = min(box1.x2, box2.x2)
    yi_s = max(box1.y1, box2.y1)
    yi_e = min(box1.y2, box2.y2)
    inter = (np.clip(xi_e - xi_s, 0, np.inf) * np.clip(yi_e - yi_s, 0, np.inf)).item()

    area1 = box1.area()
    area2 = box2.area()

    iou = inter / (area1 + area2 - inter)

    return iou


def spatial_intersection(box1: BBox, box2: BBox) -> float:
    """
    Computes intersection between two bounding boxes
    """
    xi_s = max(box1.x1, box2.x1)
    xi_e = min(box1.x2, box2.x2)
    yi_s = max(box1.y1, box2.y1)
    yi_e = min(box1.y2, box2.y2)
    inter = (np.clip(xi_e - xi_s, 0, np.inf) * np.clip(yi_e - yi_s, 0, np.inf)).item()

    return inter


def spatio_temporal_iou_response_track(rt1: ResponseTrack, rt2: ResponseTrack) -> float:
    """
    Computes tube-iou between two response track windows.
    Note: This assumes that each bbox in the list corresponds to a different
    frame. Cannot handle multiple bboxes per frame.

    Reference: https://github.com/rafaelpadilla/review_object_detection_metrics
    """
    # Map frame numbers to boxes
    boxes1_dict = {box.fno: box for box in rt1.bboxes}
    inter = 0.0
    # Find matching frame boxes and estimate iou
    for box2 in rt2.bboxes:
        box1 = boxes1_dict.get(box2.fno, None)
        if box1 is None:
            continue
        inter += spatial_intersection(box1, box2)
    # Find overall volume of the two respose tracks
    volume1 = rt1.volume()
    volume2 = rt2.volume()

    iou = inter / (volume1 + volume2 - inter)

    return iou


def spatio_temporal_iou(
    target_rt: ResponseTrack, candidate_rts: List[ResponseTrack]
) -> np.ndarray:
    """
    Computes spatio-temporal IoU between a target response track (prediction) and
    multiple candidate response tracks (ground-truth).
    """
    ious = []
    for candidate_rt in candidate_rts:
        ious.append(spatio_temporal_iou_response_track(target_rt, candidate_rt))

    return np.array(ious)


# Tracking related utils


def spatial_matches_response_track(
    pred: ResponseTrack, gt: ResponseTrack
) -> Dict[str, float]:
    """
    For each bounding box in gt, find a match in pred and measure the per-frame IoU.
    Set IoU to zero if no match is found.

    Note: This assumes that each bbox in the list corresponds to a different
    frame. Cannot handle multiple bboxes per frame.
    """
    # Map frame numbers to boxes
    gt_dict = {box.fno: box for box in gt.bboxes}
    ious = {box.fno: 0.0 for box in gt.bboxes}
    # Find matching frame boxes and estimate iou
    for pred_box in pred.bboxes:
        gt_box = gt_dict.get(pred_box.fno, None)
        if gt_box is not None:
            ious[gt_box.fno] = spatial_iou(gt_box, pred_box)
    return ious


def spatio_temporal_iou_matches(
    target_rt: ResponseTrack,
    candidate_rts: List[ResponseTrack],
) -> List[Dict[str, float]]:
    """
    For each BBox in each candidate response track (ground-truth),
    find the IoU b/w itself and a BBox from the target response track (prediction).
    In case no match is found for a particular BBox in the candidate,
    then the IoU is set to zero.
    """
    ious = []
    for candidate_rt in candidate_rts:
        ious.append(spatial_matches_response_track(target_rt, candidate_rt))
    return ious

def calculate_iou(box_g, box_p):
    """
    Calculates the spatial Intersection over Union (IoU) of two 2D bounding boxes.
    
    Bounding box format is expected to be [y_min, x_min, y_max, x_max].

    Args:
        box_g (list or np.array): Ground-truth bounding box.
        box_p (list or np.array): Predicted bounding box.

    Returns:
        float: The IoU value (between 0.0 and 1.0).
    """
    y_min_inter = max(box_g[0], box_p[0])
    x_min_inter = max(box_g[1], box_p[1])
    y_max_inter = min(box_g[2], box_p[2])
    x_max_inter = min(box_g[3], box_p[3])

    width_inter = max(0, x_max_inter - x_min_inter)
    height_inter = max(0, y_max_inter - y_min_inter)
    area_inter = width_inter * height_inter

    area_g = (box_g[2] - box_g[0]) * (box_g[3] - box_g[1])
    area_p = (box_p[2] - box_p[0]) * (box_p[3] - box_p[1])

    area_union = area_g + area_p - area_inter

    if area_union == 0:
        return 0.0
    
    iou = area_inter / area_union

    return iou

def postprocess_results(preds, threshold=0.5):
    max_probs, max_indices = torch.max(torch.sigmoid(preds['prob']), dim=-1)
    max_indices_expanded = max_indices.unsqueeze(-1).unsqueeze(-1)  # Shape: (B, T, 1, 1)
    final_results = {
        'center': torch.gather(
                            preds['center'],
                            dim=2, 
                            index=max_indices_expanded.expand(-1, -1, 1, 2) 
                        ).squeeze(2),               # Shape: (B, T, 2)
        'hw': torch.gather(
                            preds['hw'],
                            dim=2,
                            index=max_indices_expanded.expand(-1, -1, 1, 2)
                        ).squeeze(2),               # Shape: (B, T, 2)
        'bbox': torch.gather(
                            preds['bbox'],
                            dim=2, 
                            index=max_indices_expanded.expand(-1, -1, 1, 4) 
                        ).squeeze(2),               # Shape: (B, T, 4)
        'clip_with_bbox': max_probs > threshold     # Shape: (B, T)
    }
    return final_results

def spatio_temporal_IoU(preds, gt):

    batch_size, num_frames = gt['clip_with_bbox'].shape
    bbox_frames_g = [torch.nonzero(gt['clip_with_bbox'][b]).view(-1) for b in range(batch_size)]
    bbox_frames_p = [torch.nonzero(preds['clip_with_bbox'][b]).view(-1) for b in range(batch_size)]

    results = []
    count_neg = 0

    for b in range (batch_size):

        set_bbox_frames_g = set(bbox_frames_g[b].tolist())
        set_bbox_frames_p = set(bbox_frames_p[b].tolist())
        frames_intersection = set_bbox_frames_g.intersection(set_bbox_frames_p)
        frames_union = set_bbox_frames_g.union(set_bbox_frames_p)

        if not frames_union:
            if len(set_bbox_frames_g) == 0:
                count_neg += 1
            stiou = 0.0
        else:
            sum_iou = 0
            for f in frames_intersection:
                box_g = gt['clip_bbox'][b, f]
                box_p = preds['bbox'][b, f]
                iou_f = calculate_iou(box_g, box_p)
                sum_iou += iou_f
            size_union = len(frames_union)
            stiou = sum_iou / size_union
        results.append(stiou)

    return sum(results)/(len(results) - count_neg + 1e-6)

def calculate_confusion_matrix(pred_prob, gt_prob, prob_theta=0.5):
    prob_pred = (torch.sigmoid(pred_prob) > prob_theta).float()
    true_positive = (prob_pred * gt_prob).sum()
    false_positive = (prob_pred * (1 - gt_prob)).sum()
    false_negative = ((1 - prob_pred) * gt_prob).sum()

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision, recall, f1_score