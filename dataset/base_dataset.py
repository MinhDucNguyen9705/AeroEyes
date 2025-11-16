import os
import pdb

import tqdm
import random
import json

import cv2
import decord
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms as T
from dataset import dataset_utils
#from dataset.dataset_utils import normalize_bbox, recover_bbox, bbox_torchTocv2
from decord import VideoReader, cpu
import glob

split_files = {
            'train': 'vq_train.json',
            'val': 'vq_val.json',            # there is no test
            'test': 'vq_test_unannotated.json'
        }

NORMALIZE_MEAN = [int(it*255) for it in [0.485, 0.456, 0.406]]
NORMALIZE_STD = [int(it*255) for it in [0.229, 0.224, 0.225]]
    
class VisualQuery2DDataset(Dataset):
    def __init__(self, clip_params, query_params, data_paths, mode='train', transform=None, is_clip=True):
        self.clip_params = clip_params
        self.query_params = query_params
        self.data_paths = data_paths
        self.reduced_data_paths = [path.split('/')[-1] for path in self.data_paths]
        self.mode = mode
        self.is_clip = is_clip
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((self.query_params['query_size'], self.query_params['query_size'])),
                T.ToTensor()
            ])
        else:
            self.transform = transform

        if self.clip_params['padding_value'] == 'zero':
            self.padding_value = 0
        elif self.clip_params['padding_value'] == 'mean':
            self.padding_value = 0.5

        if self.mode == 'train' or self.mode == 'val':
            self.annotations_path = os.path.join(self.data_paths[0].split('/samples')[0], 'annotations/annotations.json')
            self.annotations = self._read_annotations(self.annotations_path)
            if self.mode == 'val':
                self.annotations = self.annotations * 4
            else:
                self.annotations = self.annotations * 2
            # self.response_track = {anno['video_id']: [] for anno in self.annotations}
            # for anno in self.annotations:
            #     self.response_track[anno['video_id']] += anno['response_track']
        else:
            self.annotations = None

    def _read_annotations(self, annotation_path):
        with open(annotation_path, 'r') as f:
            anno_json = json.load(f)
        self.annotations = []
        for video in anno_json:
            if video['video_id'] in self.reduced_data_paths:
                for clip_id, clip in enumerate(video['annotations']):
                    response_track_frame_ids = []
                    bboxes = clip['bboxes']
                    for bbox in bboxes:
                        response_track_frame_ids.append(int(bbox['frame']))
                    frame_id_min = min(response_track_frame_ids)
                    frame_id_max = max(response_track_frame_ids)
                    curr_anno = {
                        'video_id': video['video_id'],
                        'clip_id': clip_id, 
                        'response_track': clip['bboxes'],
                        'response_track_valid_range': [frame_id_min, frame_id_max],
                        'object_title': video['video_id'].split('_')[0],
                        'clip_fps': video.get('clip_fps', 25)
                    }
                    self.annotations.append(curr_anno)
        new_annotations = {}
        for anno in self.annotations:
            if anno['video_id'] not in new_annotations:
                new_annotations[anno['video_id']] = anno
                new_annotations[anno['video_id']]['response_track_valid_range'] = [anno['response_track_valid_range']]
            else:
                new_annotations[anno['video_id']]['response_track'] += anno['response_track']
                new_annotations[anno['video_id']]['response_track_valid_range'].append(anno['response_track_valid_range'])
        # print(new_annotations)
        self.annotations = list(new_annotations.values())
        return self.annotations

    def _get_clip_bbox(self, sample, clip_idxs, clip_h, clip_w):
        
        clip_with_bbox, clip_bbox = [], []
        response_track = sample['response_track']
        # response_track = self.response_track[sample['video_id']]
        clip_bbox_all = {}
        
        for it in response_track:
            clip_bbox_all[int(it['frame'])] = [it['y1'], it['x1'], it['y2'], it['x2']]

        # print(clip_bbox_all.keys())
        for idx in clip_idxs:
            # print(idx)
            if int(idx) in clip_bbox_all:
                clip_with_bbox.append(True)
                curr_bbox = torch.tensor(clip_bbox_all[int(idx)])
                curr_bbox_normalize = dataset_utils.normalize_bbox(curr_bbox, clip_h, clip_w)
                clip_bbox.append(curr_bbox_normalize)
            else:
                clip_with_bbox.append(False)
                clip_bbox.append(torch.tensor([0.0, 0.0, 0.00001, 0.00001]))
        clip_with_bbox = torch.tensor(clip_with_bbox).float()
        clip_bbox = torch.stack(clip_bbox, dim=0)
        return clip_with_bbox, clip_bbox

    def _get_clip_path(self, data_path):
        clip_path = glob.glob(os.path.join(data_path, '*.mp4'))[0]
        return clip_path

    def _get_query_path(self, data_path):
        query_path = glob.glob(os.path.join(data_path, 'object_images', '*.jpg'))
        return query_path
    
    def _process_clip(self, clip, clip_bbox, clip_with_bbox):
        '''
        clip: in [T,C,H,W]
        bbox: in [T,4] with torch coordinate with value range [0,1] normalized
        clip_with_bbox: in [T]
        '''
        target_size = self.clip_params['fine_size']

        t, _, h, w = clip.shape
        clip_bbox = dataset_utils.recover_bbox(clip_bbox, h, w)

        try:
            fg_idxs = torch.where(clip_with_bbox)[0].numpy().tolist()
            idx = random.choice(fg_idxs)
            frame = (clip[idx] * 255).permute(1,2,0).numpy().astype(np.uint8)
            frame = Image.fromarray(frame)
            bbox = dataset_utils.bbox_torchTocv2(clip_bbox[idx]).tolist()
            query = frame.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            query_size = self.query_params['query_size']
            query = query.resize((query_size, query_size))
            query = torch.from_numpy(np.asarray(query) / 255.0).permute(2,0,1)
        except:
            query = None

        max_size, min_size = max(h, w), min(h, w)
        pad_height = True if h < w else False
        pad_size = (max_size - min_size) // 2
        if pad_height:
            pad_input = [0, pad_size] * 2                   # for the left, top, right and bottom borders respectively
            clip_bbox[:,0] += (max_size - min_size) / 2.0   # in padded image size
            clip_bbox[:,2] += (max_size - min_size) / 2.0
        else:
            pad_input = [pad_size, 0] * 2
            clip_bbox[:,1] += (max_size - min_size) / 2.0
            clip_bbox[:,3] += (max_size - min_size) / 2.0
        
        transform_pad = T.Pad(pad_input, fill=self.padding_value)
        clip = transform_pad(clip)        # square image
        h_pad, w_pad = clip.shape[-2:]
        clip = F.interpolate(clip, size=(target_size, target_size), mode='bilinear')#.squeeze(0)
        clip_bbox = clip_bbox / float(h_pad)                # in range [0,1]

        # if self.split == 'train':
        #     clip_bbox, clip_with_bbox = self._process_bbox(clip_bbox, clip_with_bbox)
        return clip, clip_bbox, clip_with_bbox, query, h, w

    def __len__(self):
        # Return the total number of samples
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load and return a sample
        sample = self.annotations[idx]
        data_path = os.path.join('/'.join(self.data_paths[0].split('/')[:-1]), sample['video_id'])
        query_path = self._get_query_path(data_path)
        query_images = [Image.open(img_path).convert("RGB") for img_path in query_path]
        query_images = [self.transform(img) for img in query_images]
        query_images = torch.stack(query_images, dim=0)
        # query_images = Image.open(query_path[0]).convert("RGB")
        # query_images = self.transform(query_images)
        
        sample_method = self.clip_params['sampling']

        if self.is_clip:
            clip_path = self._get_clip_path(data_path)
            # if self.mode == 'train':
            #     clip, clip_idxs = read_frames_decord_random(clip_path,
            #                                                 self.clip_params['num_frames'],
            #                                                 self.clip_params['frame_interval'],
            #                                                 sample,
            #                                                 self.response_track[sample['video_id']])
            # else:
            clip, clip_idxs = read_frames_decord_balance(clip_path,
                                                        self.clip_params['num_frames'],
                                                        self.clip_params['frame_interval'],
                                                        sample,
                                                        sampling=sample_method)
        else:
            clip_path = glob.glob(os.path.join(data_path, 'video/*.jpg')) + glob.glob(os.path.join(data_path, 'video/*.png')) + glob.glob(os.path.join(data_path, 'video/*.jpeg'))
            clip_path.sort()
            clip_idxs = sample_frames_balance(self.clip_params['num_frames'],
                                            self.clip_params['frame_interval'], 
                                            sample, 
                                            sampling='rand')
            clip_idxs = [min(it, len(clip_path)-1) for it in clip_idxs]
            sample_clip_path = [clip_path[i] for i in clip_idxs]
            clip = [Image.open(img_path).convert("RGB") for img_path in sample_clip_path]
            clip = [T.ToTensor()(img) for img in clip]
            clip = torch.stack(clip, dim=0)

        # print(clip_idxs)
        # print(sample['response_track'])
        clip_h, clip_w = clip.shape[-2], clip.shape[-1]

        clip_with_bbox, clip_bbox = self._get_clip_bbox(sample, clip_idxs, clip_h, clip_w)
        # print(clip_with_bbox)
        clip, clip_bbox, clip_with_bbox, query, clip_h, clip_w = self._process_clip(clip, clip_bbox, clip_with_bbox)
        
        results = {
            'object_title': sample['object_title'],
            'clip': clip,    # [num_frame, C, H, W]
            'clip_with_bbox': clip_with_bbox,
            'clip_bbox': clip_bbox.float(),
            'clip_idxs': torch.tensor(clip_idxs),
            'query_images': query_images,
            'clip_h': clip_h,
            'clip_w': clip_w,
            # 'query': query
        }

        return results

class TestDataset(Dataset):
    def __init__(self, clip_params, query_params, video_paths, transform=None, pad_last=True):
        self.video_paths = video_paths
        self.num_frames = clip_params['num_frames']
        self.frame_interval = clip_params['frame_interval']
        self.pad_last = pad_last
        self.clip_params = clip_params
        self.query_params = query_params
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((self.query_params['query_size'], self.query_params['query_size'])),
                T.ToTensor()
            ])
        else:
            self.transform = transform
            
        self.clips = self._index_clips()

    def _index_clips(self):
        clips = []
        for vid_path in self.video_paths:
            query_path = glob.glob(os.path.join('/'.join(vid_path.split('/')[:-1]), 'object_images/*.jpg'))
            vr = VideoReader(vid_path, ctx=cpu(0))
            total = len(vr)
            last_start = 0
            for start in range(0, total - self.num_frames + 1, self.frame_interval):
                clips.append((vid_path, query_path, start, total))
                last_start = start
            if self.pad_last and last_start + self.num_frames + self.frame_interval > total:
                clips.append((vid_path, query_path, last_start+self.num_frames, total))
        return clips

    def process_clip(self, clip):

        target_size = self.clip_params['fine_size']        
        t, _, h, w = clip.shape
        max_size, min_size = max(h, w), min(h, w)
        pad_height = True if h < w else False
        pad_size = (max_size - min_size) // 2
        if pad_height:
            pad_input = [0, pad_size] * 2                   # for the left, top, right and bottom borders respectively
        else:
            pad_input = [pad_size, 0] * 2
        
        transform_pad = T.Pad(pad_input, fill=0)
        clip = transform_pad(clip)        # square image
        h_pad, w_pad = clip.shape[-2:]
        clip = F.interpolate(clip, size=(target_size, target_size), mode='bilinear')#.squeeze(0)

        return clip, h, w

    def get_clip_with_idxs(self, vid_path, start):
        
        vr = VideoReader(vid_path, ctx=cpu(0))
        total = len(vr)
        end = start + self.num_frames

        if end > total:
            indices = list(range(start, total))
            clip = vr.get_batch(indices).permute(0, 3, 1, 2).float() / 255.0
            pad_count = self.num_frames - clip.shape[0]
            pad = torch.zeros((pad_count, *clip.shape[1:]), dtype=clip.dtype)
            clip = torch.cat([clip, pad], dim=0)
        else:
            clip = vr.get_batch(range(start, end)).permute(0, 3, 1, 2).float() / 255.0
        
        return clip
    
    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        
        vid_path, query_path, start, total = self.clips[idx]
        clip = self.get_clip_with_idxs(vid_path, start)
        clip, h, w = self.process_clip(clip)
            
        query_images = [Image.open(img_path).convert("RGB") for img_path in query_path]
        query_images = [self.transform(img) for img in query_images]
        query_images = torch.stack(query_images, dim=0)
        # query_images = Image.open(query_path[0]).convert("RGB")
        # query_images = self.transform(query_images)
        
        results = {
            'video_id': vid_path.split('/')[-2],
            'clip': clip,
            'clip_idxs': torch.arange(start, start+self.num_frames),
            'query_images': query_images,
            'clip_h': h,
            'clip_w': w,
            'total_frames': total
        }
        
        return results  # shape [T, C, H, W]

def sample_frames_balance(num_frames, frame_interval, sample, sampling='rand'):
    '''
    sample clips with balanced negative and positive samples
    params:
        num_frames: total number of frames to sample
        query_frame: query time index
        frame_interval: frame interval, where value 1 is for no interval (consecutive frames)
        sample: data annotations
        sampling: only effective for frame_interval larger than 1
    return: 
        frame_idxs: length [num_frames]
    '''
    required_len = (num_frames - 1) * frame_interval + 1
    valid_idx = np.random.choice(range(len(sample["response_track_valid_range"])))
    anno_valid_idx_range = sample["response_track_valid_range"][valid_idx]
    anno_len = anno_valid_idx_range[1] - anno_valid_idx_range[0] + 1
    
    if anno_len <= required_len:
        if anno_len < required_len:
            num_valid = anno_len // frame_interval
        else:
            num_valid = num_frames
        num_invalid = num_frames - num_valid
        if anno_valid_idx_range[1] < required_len:
            idx_start = random.choice(range(anno_valid_idx_range[0])) if anno_valid_idx_range[0] > 0 else 0
            idx_end = idx_start + required_len
        else:
            num_prior = random.choice(range(num_invalid)) if num_invalid != 0 else 0
            num_post = num_invalid - num_prior
            idx_start = anno_valid_idx_range[0] - frame_interval * num_prior
            idx_end = anno_valid_idx_range[1] + frame_interval * num_post + 1
        intervals = np.linspace(start=idx_start, stop=idx_end, num=num_frames+1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1]))
        if sampling == 'rand':
            frame_idxs_pos = [random.choice(range(x[0], x[1])) for x in ranges]
        elif sampling == 'uniform':
            frame_idxs_pos = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        num_addition = anno_len - required_len
        start = random.choice(range(num_addition))
        frame_idxs_pos = [anno_valid_idx_range[0] + start + it for it in range(num_frames)]
    return frame_idxs_pos

decord.bridge.set_bridge("torch")

def read_frames_decord_balance(video_path, num_frames, frame_interval, sample, sampling='rand'):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    origin_fps = int(video_reader.get_avg_fps())
    gt_fps = int(sample.get('clip_fps', 25))
    down_rate = origin_fps // gt_fps if gt_fps > 0 else 1
    frame_idxs = sample_frames_balance(num_frames, frame_interval, sample, sampling)      # downsampled fps idxs, used to get bbox annotation
    frame_idxs_origin = [min(it * down_rate, vlen - 1) for it in frame_idxs]        # origin clip fps frame idxs
    #video_reader.skip_frames(1)
    frames = video_reader.get_batch(frame_idxs_origin)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


def get_bbox_from_data(data):
    # BoxMode.XYXY_ABS
    return [data["x"], data["y"], data["x"] + data["width"], data["y"] + data["height"]]

def get_video_len(video_path):
    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen

video_reader_dict = {
    'decord_balance': read_frames_decord_balance,
}