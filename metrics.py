import torch

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

    for b in range (batch_size):

        set_bbox_frames_g = set(bbox_frames_g[b].tolist())
        set_bbox_frames_p = set(bbox_frames_p[b].tolist())
        frames_intersection = set_bbox_frames_g.intersection(set_bbox_frames_p)
        frames_union = set_bbox_frames_g.union(set_bbox_frames_p)

        if not frames_union:
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

    return sum(results)/len(results)