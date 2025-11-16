import cv2
import json
import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def gen_gaussian2d(shape, sigma=1):
    h, w = [_ // 2 for _ in shape]
    y, x = np.ogrid[-h : h + 1, -w : w + 1]
    gaussian = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    return gaussian

def draw_gaussian(density, center, radius, k=1, delte=6, overlap="add"):
    """
    This is your draw_gaussian function.
    I have removed the print() statements to avoid log spam.
    """
    diameter = 2 * radius + 1
    gaussian = gen_gaussian2d((diameter, diameter), sigma=diameter / delte)
    x, y = center
    
    height, width = density.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    if overlap == "max":
        masked_density = density[int(y - top) : int(y + bottom), int(x - left) : int(x + right)]
        masked_gaussian = gaussian[
            int(radius - top) : int(radius + bottom), int(radius - left) : int(radius + right)
        ]
        if masked_density.shape != masked_gaussian.shape:
            r_h, r_w = masked_density.shape
            masked_gaussian = masked_gaussian[:r_h, :r_w]

        np.maximum(masked_density, masked_gaussian * k, out=masked_density)
    
    elif overlap == "add":
        density[y - top : y + bottom, x - left : x + right] += gaussian[
            radius - top : radius + bottom, radius - left : radius + right
        ]
    else:
        raise NotImplementedError

def _min_dis_global(points):
    """
    points: m x 2, m x [x, y]
    """
    dis_min = float("inf")
    for point in points:
        point = point[None, :]  # 2 -> 1 x 2
        dis = np.sqrt(np.sum((points - point) ** 2, axis=1))
        dis = sorted(dis)[1]
        if dis_min > dis:
            dis_min = dis
    return dis_min

def points2density(points, image_size, radius_backup=None):
    """
    This is your points2density function.
    I've modified it to accept 'image_size' as an argument.
    
    points: m x 2, m x [x, y]
    image_size: (height, width)
    """
    num_points = points.shape[0]
    density = np.zeros(image_size, dtype=np.float32)
    
    if num_points == 0:
        return np.zeros(image_size, dtype=np.float32)
    elif num_points == 1:
        radius = radius_backup
    else:
        radius = min(int(_min_dis_global(points)), radius_backup)
    
    radius = max(1, int(radius))

    for point in points:
        draw_gaussian(density, point, radius, overlap="max")
    
    return density

def process_videos(annotation_path, video_dir, video_filename_in_folder):
    with open(annotation_path, 'r') as f:
        all_annotations = json.load(f)

    video_lookup = {}
    for video_data in all_annotations:
        video_id = video_data['video_id']
        video_lookup[video_id] = []
        if 'annotations' in video_data and video_data['annotations']:
            for annotation_track in video_data['annotations']:
                video_lookup[video_id].extend(annotation_track.get('bboxes', []))
    
    print(f"Found annotations for {len(video_lookup)} videos.")

    video_subfolders = [d for d in os.listdir(video_dir) 
                            if os.path.isdir(os.path.join(video_dir, d))]
        
    print(f"Found {len(video_subfolders)} video subfolders in {video_dir}.")

    for video_id in tqdm(video_subfolders, desc="Processing videos"):
        
        if video_id not in video_lookup:
            print(f"Warning: Found folder '{video_id}' but no matching annotations. Skipping.")
            continue
        
        video_path = os.path.join(video_dir, video_id, video_filename_in_folder)

        if not os.path.exists(video_path):
            print(f"Warning: Annotation found for '{video_id}', but video file not found at {video_path}. Skipping.")
            continue

        frame_to_boxes = {}
        for box in video_lookup[video_id]:
            frame_num = box['frame']
            if frame_num not in frame_to_boxes:
                frame_to_boxes[frame_num] = []
            frame_to_boxes[frame_num].append(box)

        cap = cv2.VideoCapture(video_path)

        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        image_size = (frame_height, frame_width)

        video_output_dir = os.path.join(video_dir, video_id, "density_map")
        os.makedirs(video_output_dir, exist_ok=True)

        for frame_num in range(total_frames):
            
            boxes_in_frame = frame_to_boxes.get(frame_num, [])
            
            if not boxes_in_frame:
                density_map = np.zeros(image_size, dtype=np.float32)
            else:
                points = []
                radius_backups = []
                
                for box in boxes_in_frame:
                    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                    w, h = x2 - x1, y2 - y1
                    
                    if w <= 0 or h <= 0:
                        continue
                        
                    points.append([(x1 + x2) // 2, (y1 + y2) // 2])
                    radius_backups.append((w + h) // 2)

                if not points:
                    density_map = np.zeros(image_size, dtype=np.float32)
                else:
                    points_array = np.array(points)
                    avg_radius_backup = max(1, int(np.mean(radius_backups)))
                    density_map = points2density(
                        points_array, 
                        image_size, 
                        radius_backup=avg_radius_backup
                    )
            
            save_path = os.path.join(video_output_dir, f"{frame_num:05d}.png")
            plt.imsave(save_path, density_map, cmap='jet', vmin=0, vmax=1)

        cap.release()

    print("\nAll videos processed.")

ANNOTATION_FILE = "D:/OneDrive - Hanoi University of Science and Technology/Domain Basics/Introduction to Deep Learning/Data/train/annotations/annotations.json"
VIDEO_ROOT_DIR  = "D:/OneDrive - Hanoi University of Science and Technology/Domain Basics/Introduction to Deep Learning/Data/train/samples"
VIDEO_FILENAME_IN_FOLDER = "drone_video.mp4" 

if __name__ == "__main__":
    process_videos(
        annotation_path=ANNOTATION_FILE,
        video_dir=VIDEO_ROOT_DIR,
        video_filename_in_folder=VIDEO_FILENAME_IN_FOLDER
    )