import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset.dataset_utils import recover_bbox, NORMALIZE_MEAN, NORMALIZE_STD, unnormalize, process_data
import wandb
from torch.utils.data import DataLoader, Subset
import random
from metrics.utils import calculate_iou, postprocess_results
from dataset import dataset_utils

def visualization(config, model, dataloader, epoch, device, num_samples=4):

    num_samples = min(dataloader.batch_size, num_samples)
    model.eval()
    dataset = dataloader.dataset
    random_indices = torch.randperm(len(dataset))[:num_samples]
    subset = torch.utils.data.Subset(dataset, random_indices)
    
    # new dataloader (no shuffle, since indices are already random)
    random_loader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=False)
    batch = next(iter(random_loader))
    batch = dataset_utils.replicate_sample_for_hnm(batch)
    batch = process_data(config, batch, split='val', device=device)

    with torch.no_grad():
        clips = batch['clip'].to(device)
        queries = batch['query'].to(device)
        output = model(clips, queries, training=False, fix_backbone=True)
        final_output = postprocess_results(output)
        
    fig, ax = plt.subplots(num_samples, 5, figsize=(10, num_samples*3))
    num_frames = batch['clip'].shape[1]
    batch_idx = np.random.choice(batch['clip'].shape[0], num_samples, replace=False)

    for i in range (num_samples):
        frame = np.random.randint(0, num_frames)
        clip = batch['clip'][batch_idx[i], frame].permute(1, 2, 0).cpu().numpy()
        clip = unnormalize(clip, NORMALIZE_MEAN, NORMALIZE_STD)
        h, w, _ = clip.shape
        ax[i, 0].imshow(clip)
        ax[i, 0].axis('off')
        ax[i, 0].set_title(f"{batch['object_title'][batch_idx[i]]}")
        if batch['clip_with_bbox'][batch_idx[i], frame] == 1:
            gt_bbox = batch['clip_bbox'][batch_idx[i], frame]
            gt_bbox = recover_bbox(gt_bbox, h, w)
            rect = plt.Rectangle((gt_bbox[1], gt_bbox[0]), (gt_bbox[3]-gt_bbox[1]), (gt_bbox[2]-gt_bbox[0]), linewidth=1, edgecolor='r', facecolor='none')
            ax[i, 0].add_patch(rect)
        ax[i, 0].set_title('GT')
        ax[i, 1].imshow(clip)
        if final_output['clip_with_bbox'][batch_idx[i], frame] == 1:
            bbox = final_output['bbox'][batch_idx[i], frame]
            bbox = recover_bbox(bbox, h, w)
            rect = plt.Rectangle((bbox[1], bbox[0]), (bbox[3]-bbox[1]), (bbox[2]-bbox[0]), linewidth=1, edgecolor='r', facecolor='none')
            ax[i, 1].add_patch(rect)
        if batch['clip_with_bbox'][batch_idx[i], frame] == 1 and final_output['clip_with_bbox'][batch_idx[i], frame] == 1:
            iou = calculate_iou(batch['clip_bbox'][batch_idx[i], frame], final_output['bbox'][batch_idx[i], frame])
        else:
            iou = 0.0
        ax[i, 1].axis('off')
        ax[i, 1].set_title(f'Predicted, IoU = {iou: .2f}')
        ax[i, 2].imshow(unnormalize(batch['query_images'][batch_idx[i], 0].permute(1,2,0).cpu().numpy(), NORMALIZE_MEAN, NORMALIZE_STD))
        ax[i, 2].axis('off')
        ax[i, 2].set_title('Query Image 1')
        ax[i, 3].imshow(unnormalize(batch['query_images'][batch_idx[i], 1].permute(1,2,0).cpu().numpy(), NORMALIZE_MEAN, NORMALIZE_STD))
        ax[i, 3].axis('off')
        ax[i, 3].set_title('Query Image 2')
        ax[i, 4].imshow(unnormalize(batch['query_images'][batch_idx[i], 2].permute(1,2,0).cpu().numpy(), NORMALIZE_MEAN, NORMALIZE_STD))
        ax[i, 4].axis('off')
        ax[i, 4].set_title('Query Image 3')
    
    plt.tight_layout()
    wandb.log({f"Epoch {epoch}": wandb.Image(fig)})
    plt.close(fig)