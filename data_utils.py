def normalize_bbox(bbox, h, w):
    '''
    bbox torch tensor in shape [4] or [...,4], under torch axis
    '''
    bbox_cp = bbox.clone()
    if len(bbox.shape) > 1: # [N,4]
        bbox_cp[..., 0] /= h
        bbox_cp[..., 1] /= w
        bbox_cp[..., 2] /= h
        bbox_cp[..., 3] /= w
        return bbox_cp
    else:
        return torch.tensor([bbox_cp[0]/h, bbox_cp[1]/w, bbox_cp[2]/h, bbox_cp[3]/w])

def recover_bbox(bbox, h, w):
    '''
    bbox torch tensor in shape [4] or [...,4], under torch axis
    '''
    bbox_cp = bbox.clone()
    if len(bbox.shape) > 1: # [N,4]
        bbox_cp[..., 0] *= h
        bbox_cp[..., 1] *= w
        bbox_cp[..., 2] *= h
        bbox_cp[..., 3] *= w
        return bbox_cp
    else:
        return torch.tensor([bbox_cp[0]*h, bbox_cp[1]*w, bbox_cp[2]*h, bbox_cp[3]*w])

def bbox_torchTocv2(bbox):
    '''
    torch, idx 0/2 for height, 1/3 for width (x,y,x,y)
    cv2: idx 0/2 for width, 1/3 for height (y,x,y,x)
    bbox torch tensor in shape [4] or [...,4], under torch axis
    '''
    bbox_cp = bbox.clone()
    if len(bbox.shape) > 1:
        bbox_x1 = bbox_cp[...,0].unsqueeze(-1)
        bbox_y1 = bbox_cp[...,1].unsqueeze(-1)
        bbox_x2 = bbox_cp[...,2].unsqueeze(-1)
        bbox_y2 = bbox_cp[...,3].unsqueeze(-1)
        return torch.cat([bbox_y1, bbox_x1, bbox_y2, bbox_x2], dim=-1)
    else:
        return torch.tensor([bbox_cp[1], bbox_cp[0], bbox_cp[3], bbox_cp[2]])

import torch

def recover_boxes_to_original(bboxes_norm_sq, orig_h, orig_w):
    """
    Invert your padding+normalize pipeline.
    Args:
        bboxes_norm_sq: Tensor [T,4] in [0,1], normalized by the padded square size
                        (this is what __getitem__ returns as 'clip_bbox').
        orig_h, orig_w: integers, original frame height/width (before pad/resize).
    Returns:
        b_abs: Tensor [T,4] absolute pixel coords on the original frame (y1,x1,y2,x2).
        b_norm_orig: Tensor [T,4] normalized by (orig_h, orig_w).
    """
    if bboxes_norm_sq.numel() == 0:
        return bboxes_norm_sq, bboxes_norm_sq

    # Size of the padded square before resize
    h_pad = w_pad = max(orig_h, orig_w)

    # Compute exact padding put on each side (handle odd differences)
    if orig_h < orig_w:
        # padded in height: pad equally (floor on top, ceil on bottom)
        total = orig_w - orig_h
        pad_top  = total // 2
        pad_bot  = total - pad_top
        pad_left = pad_right = 0
    else:
        # padded in width
        total = orig_h - orig_w
        pad_left = total // 2
        pad_right = total - pad_left
        pad_top = pad_bot = 0

    # 1) un-normalize from square
    b = bboxes_norm_sq.clone() * float(h_pad)  # square so h_pad == w_pad

    # 2) remove padding offsets
    # y's: indices 0,2 ; x's: indices 1,3
    b[:, [0, 2]] -= pad_top
    b[:, [1, 3]] -= pad_left

    # 3) clip to original image bounds
    b[:, [0, 2]] = b[:, [0, 2]].clamp(0, orig_h - 1)
    b[:, [1, 3]] = b[:, [1, 3]].clamp(0, orig_w - 1)

    # 4) (optional) normalize by original H/W
    b_norm_orig = b.clone()
    b_norm_orig[:, [0, 2]] /= float(orig_h)
    b_norm_orig[:, [1, 3]] /= float(orig_w)

    return b, b_norm_orig