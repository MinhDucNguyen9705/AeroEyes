import os
import sys
import glob
import tempfile
import subprocess

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms as T
from decord import VideoReader, cpu
import kornia

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config
from model.corr_clip_spatial_transformer2_anchor_2heads_hnm import ClipMatcher
from metrics.utils import postprocess_results


# ============== Configuration ==============
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", os.path.join(SCRIPT_DIR, "cpt_best_iou.pth.tar"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = config.dataset.clip_num_frames  # 30
FRAME_INTERVAL = config.dataset.frame_interval  # 5
CLIP_SIZE = config.dataset.clip_size_fine  # 448
QUERY_SIZE = config.dataset.query_size  # 448

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


# ============== Model Loading ==============
def load_model():
    """Load the ClipMatcher model with pretrained weights."""
    print(f"Loading model on {DEVICE}...")
    model = ClipMatcher(config).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
    else:
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}")
    
    model.eval()
    return model


# Global model instance
MODEL = None


def get_model():
    """Lazy load model on first use."""
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL


# ============== Data Processing ==============
def process_clip(clip_tensor):
    """
    Pad and resize clip to square target size.
    Args:
        clip_tensor: [T, C, H, W] float tensor in [0, 1]
    Returns:
        processed_clip: [T, C, CLIP_SIZE, CLIP_SIZE]
        orig_h, orig_w: original dimensions
    """
    t, c, h, w = clip_tensor.shape
    orig_h, orig_w = h, w
    
    # Pad to square
    max_size = max(h, w)
    pad_h = (max_size - h) // 2
    pad_w = (max_size - w) // 2
    
    # Pad: (left, right, top, bottom)
    clip_padded = F.pad(clip_tensor, (pad_w, max_size - w - pad_w, pad_h, max_size - h - pad_h), value=0)
    
    # Resize to target size
    clip_resized = F.interpolate(clip_padded, size=(CLIP_SIZE, CLIP_SIZE), mode='bilinear', align_corners=False)
    
    return clip_resized, orig_h, orig_w


def normalize_clip(clip_tensor):
    """Apply ImageNet normalization using kornia."""
    return kornia.enhance.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(clip_tensor)


def recover_boxes_to_original(bboxes_norm, orig_h, orig_w):
    """
    Convert normalized bounding boxes back to original pixel coordinates.
    
    Args:
        bboxes_norm: Tensor [T, 4] in [0, 1], normalized by the padded square size.
                     Format: [y1, x1, y2, x2]
        orig_h, orig_w: Original frame dimensions before padding/resizing.
    
    Returns:
        bboxes_abs: Tensor [T, 4] in absolute pixel coordinates [y1, x1, y2, x2]
    """
    if bboxes_norm.numel() == 0:
        return bboxes_norm
    
    # The padded square size
    h_pad = w_pad = max(orig_h, orig_w)
    
    # Calculate padding offsets
    if orig_h < orig_w:
        # Height was padded
        pad_top = (orig_w - orig_h) // 2
        pad_left = 0
    else:
        # Width was padded  
        pad_top = 0
        pad_left = (orig_h - orig_w) // 2
    
    # Clone and scale from [0,1] to padded square size
    bboxes = bboxes_norm.clone().float() * float(h_pad)
    
    # Remove padding offsets
    # y coords: indices 0, 2
    # x coords: indices 1, 3
    bboxes[:, [0, 2]] -= pad_top
    bboxes[:, [1, 3]] -= pad_left
    
    # Clamp to original image bounds
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clamp(0, orig_h - 1)
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clamp(0, orig_w - 1)
    
    return bboxes


def load_video_clip(video_path, start_frame, num_frames):
    """
    Load a clip of frames from video.
    
    Args:
        video_path: Path to video file
        start_frame: Starting frame index
        num_frames: Number of frames to load
    
    Returns:
        clip: [T, C, H, W] tensor in [0, 1]
        frame_indices: List of actual frame indices loaded
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    end_frame = min(start_frame + num_frames, total_frames)
    actual_frames = end_frame - start_frame
    
    indices = list(range(start_frame, end_frame))
    frames = vr.get_batch(indices)
    
    # Convert to tensor [T, C, H, W]
    if hasattr(frames, 'asnumpy'):
        frames = frames.asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    
    # Pad if needed
    if actual_frames < num_frames:
        pad_count = num_frames - actual_frames
        pad_frames = torch.zeros((pad_count, *frames.shape[1:]), dtype=frames.dtype)
        frames = torch.cat([frames, pad_frames], dim=0)
        indices.extend([-1] * pad_count)  # Mark padded frames
    
    return frames, indices


def load_query_images(image_paths):
    """
    Load and process query images.
    
    Args:
        image_paths: List of image file paths
    
    Returns:
        query_tensor: [N, C, H, W] tensor
    """
    transform = T.Compose([
        T.Resize((QUERY_SIZE, QUERY_SIZE)),
        T.ToTensor()
    ])
    
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img)
        images.append(img_tensor)
    
    return torch.stack(images, dim=0)


# ============== Inference ==============
@torch.no_grad()
def run_inference(video_path, query_images, threshold=0.5, progress=gr.Progress()):
    """
    Run inference on a video with query images.
    
    Args:
        video_path: Path to input video
        query_images: List of query image paths or PIL Images
        threshold: Detection confidence threshold
        progress: Gradio progress tracker
    
    Returns:
        detections: Dict mapping frame_idx -> bbox (y1, x1, y2, x2)
    """
    model = get_model()
    
    # Load video info
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    
    # Prepare query images
    if isinstance(query_images, list) and len(query_images) > 0:
        query_list = []
        for q in query_images:
            if isinstance(q, tuple):
                # Gradio Gallery returns (image, caption) tuples
                q = q[0]
            if isinstance(q, str):
                query_list.append(q)
            elif isinstance(q, np.ndarray):
                # Save temp image
                temp_path = tempfile.mktemp(suffix=".jpg")
                Image.fromarray(q).save(temp_path)
                query_list.append(temp_path)
            elif isinstance(q, Image.Image):
                temp_path = tempfile.mktemp(suffix=".jpg")
                q.save(temp_path)
                query_list.append(temp_path)
        
        query_tensor = load_query_images(query_list)
    else:
        raise ValueError("No query images provided")
    
    # Normalize query images
    query_tensor = normalize_clip(query_tensor)
    query_tensor = query_tensor.unsqueeze(0).to(DEVICE)  # [1, N, C, H, W]
    
    all_detections = {}
    
    # Process video in clips
    num_clips = (total_frames + NUM_FRAMES - 1) // NUM_FRAMES
    
    for clip_idx in progress.tqdm(range(num_clips), desc="Processing video"):
        start_frame = clip_idx * NUM_FRAMES
        
        # Load and process clip
        clip, frame_indices = load_video_clip(video_path, start_frame, NUM_FRAMES)
        clip_processed, orig_h, orig_w = process_clip(clip)
        clip_normalized = normalize_clip(clip_processed)
        clip_batch = clip_normalized.unsqueeze(0).to(DEVICE)  # [1, T, C, H, W]
        
        # Run model
        output = model(clip_batch, query_tensor, training=False, fix_backbone=True)
        results = postprocess_results(output, threshold=threshold)
        
        # Extract detections
        has_bbox = results['clip_with_bbox'][0]  # [T]
        bboxes = results['bbox'][0]  # [T, 4]
        
        # Find frames with detections
        det_indices = torch.where(has_bbox)[0]
        
        if det_indices.numel() > 0:
            det_bboxes = bboxes[det_indices]  # [N_det, 4]
            
            # Convert to original coordinates
            det_bboxes_orig = recover_boxes_to_original(det_bboxes.cpu(), orig_h, orig_w)
            
            for i, det_idx in enumerate(det_indices.tolist()):
                frame_idx = frame_indices[det_idx]
                if frame_idx >= 0:  # Skip padded frames
                    bbox = det_bboxes_orig[i].tolist()
                    all_detections[frame_idx] = bbox
    
    return all_detections, total_frames, orig_h, orig_w


def create_output_video(video_path, detections, output_path, orig_h, orig_w, progress=gr.Progress()):
    """
    Create output video with bounding boxes drawn.
    
    Args:
        video_path: Input video path
        detections: Dict mapping frame_idx -> bbox (y1, x1, y2, x2)
        output_path: Output video path
        orig_h, orig_w: Original frame dimensions
        progress: Gradio progress tracker
    
    Returns:
        output_path: Path to output video
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temp AVI file first (more reliable encoding)
    temp_avi = output_path.replace('.mp4', '_temp.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_avi, fourcc, fps, (width, height))
    
    frame_idx = 0
    for _ in progress.tqdm(range(total_frames), desc="Creating output video"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw bounding box if detection exists
        if frame_idx in detections:
            y1, x1, y2, x2 = detections[frame_idx]
            y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
            
            # Draw rectangle (BGR: green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw label
            label = f"Frame {frame_idx}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    # Convert to H.264 MP4 for browser compatibility
    try:
        cmd = [
            'ffmpeg', '-y', '-i', temp_avi,
            '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '23', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(temp_avi)
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}, using AVI")
        os.rename(temp_avi, output_path.replace('.mp4', '.avi'))
        output_path = output_path.replace('.mp4', '.avi')
    
    return output_path


# ============== Gradio Interface ==============
def process_video(video_file, query_gallery, threshold, progress=gr.Progress()):
    """
    Main processing function for Gradio interface.
    
    Args:
        video_file: Uploaded video file path
        query_gallery: Gallery of query images
        threshold: Detection threshold
        progress: Gradio progress tracker
    
    Returns:
        output_video: Path to output video with detections
        status: Status message
    """
    if video_file is None:
        return None, "❌ Please upload a video file."
    
    if query_gallery is None or len(query_gallery) == 0:
        return None, "❌ Please upload at least one query image."
    
    try:
        # Run inference
        progress(0.1, desc="Running inference...")
        detections, total_frames, orig_h, orig_w = run_inference(
            video_file, query_gallery, threshold=threshold, progress=progress
        )
        
        num_detections = len(detections)
        
        if num_detections == 0:
            return None, f"⚠️ No detections found in {total_frames} frames. Try lowering the threshold."
        
        # Create output video
        progress(0.7, desc="Creating output video...")
        output_dir = tempfile.mkdtemp()
        output_path = os.path.join(output_dir, "output_with_boxes.mp4")
        
        output_video = create_output_video(
            video_file, detections, output_path, orig_h, orig_w, progress=progress
        )
        
        progress(1.0, desc="Done!")
        status = f"✅ Found {num_detections} detections across {total_frames} frames."
        
        return output_video, status
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


def create_demo():
    """Create the Gradio demo interface."""
    
    with gr.Blocks(title="AeroEyes - Visual Object Tracking", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚁 AeroEyes - Visual Object Tracking Demo
        
        Upload a drone video and query images of the object you want to track.
        The model will detect and localize the object throughout the video.
        
        ## Instructions:
        1. **Upload Video**: Upload the drone video (MP4 format recommended)
        2. **Upload Query Images**: Upload one or more reference images of the target object
        3. **Adjust Threshold**: Lower values = more detections (may include false positives)
        4. **Click Process**: Wait for the model to process the video
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="📹 Input Video", format="mp4")
                query_gallery = gr.Gallery(
                    label="🎯 Query Images (object to track)",
                    columns=3,
                    height=200,
                    object_fit="contain"
                )
                query_upload = gr.File(
                    label="Upload query images",
                    file_count="multiple",
                    file_types=["image"]
                )
                threshold_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="🎚️ Detection Threshold"
                )
                process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                video_output = gr.Video(label="📤 Output Video with Detections")
                status_text = gr.Textbox(label="📊 Status", interactive=False)
        
        # Handle query image uploads
        def update_gallery(files):
            if files is None:
                return []
            return [f.name for f in files]
        
        query_upload.change(
            fn=update_gallery,
            inputs=[query_upload],
            outputs=[query_gallery]
        )
        
        # Process button click
        process_btn.click(
            fn=process_video,
            inputs=[video_input, query_gallery, threshold_slider],
            outputs=[video_output, status_text]
        )
        
        gr.Markdown("""
        ---
        ### Notes:
        - **Model**: ClipMatcher with DINOv2 backbone
        - **Input Size**: 448x448 (automatically resized)
        - **Frames per Clip**: 30 frames processed at a time
        - Processing time depends on video length and GPU availability
        """)
    
    return demo


# ============== Main ==============
if __name__ == "__main__":
    print("Starting AeroEyes Gradio Demo...")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        show_error=True
    )
