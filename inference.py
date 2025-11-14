import os
import glob
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.dataset_utils import get_dataset, recover_bbox, recover_boxes_to_original
from metrics import postprocess_results
from collections import defaultdict

def inference(model, test_path, output_path):
    # Discover videos
    video_paths = glob.glob(f'{test_path}/**/*.mp4', recursive=True)

    # Dataset / loader
    test_dataset = TestDataset(clip_params, query_params, video_paths)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Results structure and per-video running state
    results = {}  # video_id -> {"video_id": id, "detections": [ {"bboxes":[...]} , ... ]}
    state = defaultdict(lambda: {"last_idx": None, "bboxes": []})  # per-video temp state

    def ensure_video(video_id: str):
        if video_id not in results:
            results[video_id] = {"video_id": video_id, "detections": []}

    def flush(video_id: str):
        st = state[video_id]
        if st["bboxes"]:
            results[video_id]["detections"].append({"bboxes": st["bboxes"]})
            st["bboxes"] = []

    def append_bbox(video_id: str, frame: int, x1: int, y1: int, x2: int, y2: int):
        st = state[video_id]
        last = st["last_idx"]
        if (last is None) or (frame != last + 1):
            # new run -> flush previous run if any
            if st["bboxes"]:
                results[video_id]["detections"].append({"bboxes": st["bboxes"]})
                st["bboxes"] = []
        st["bboxes"].append({"frame": int(frame), "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
        st["last_idx"] = int(frame)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = process_data(config, batch, split='val', device=device)
            clips = batch['clip'].to(device, non_blocking=True)
            queries = batch['query_images'].to(device, non_blocking=True)

            output = model(clips, queries, training=False, fix_backbone=True)
            final_output = postprocess_results(output)

            batch_size = clips.shape[0]

            for b in range(batch_size):
                # video id as string key
                vid = batch['video_id'][b]
                if isinstance(vid, (list, tuple)):
                    vid = vid[0]
                if isinstance(vid, torch.Tensor):
                    vid = str(vid.item())
                else:
                    vid = str(vid)
                ensure_video(vid)

                # sizes
                clip_h = batch['clip_h'][b]
                clip_w = batch['clip_w'][b]
                if isinstance(clip_h, torch.Tensor):
                    clip_h = int(clip_h.item())
                if isinstance(clip_w, torch.Tensor):
                    clip_w = int(clip_w.item())

                total_frames = batch.get('total_frames', None)
                if total_frames is not None:
                    tf = batch['total_frames'][b]
                    if isinstance(tf, torch.Tensor):
                        tf = int(tf.item())
                    else:
                        tf = int(tf)
                else:
                    tf = None

                # positions with boxes
                has_box = final_output['clip_with_bbox'][b]  # bool tensor over time
                if has_box.numel() == 0:
                    continue
                t_pos = torch.where(has_box)[0]
                if t_pos.numel() == 0:
                    continue

                # frame indices for those positions
                clip_idxs = batch['clip_idxs'][b]
                if clip_idxs.device != t_pos.device:
                    clip_idxs = clip_idxs.to(t_pos.device)

                pos_frame_idxs = clip_idxs[t_pos]                  # (N,)
                pos_bboxes = final_output['bbox'][b, t_pos, :]     # (N, 4)

                # sort by frame to ensure correct consecutiveness
                order = torch.argsort(pos_frame_idxs)
                pos_frame_idxs = pos_frame_idxs[order]
                pos_bboxes = pos_bboxes[order]

                # append boxes for this video
                for i in range(pos_frame_idxs.shape[0]):
                    frame = int(pos_frame_idxs[i].item())
                    if (tf is not None) and (frame >= tf):
                        continue

                    # recover to original coords; expected (1,4) -> [[y1,x1,y2,x2]]
                    rec = recover_boxes_to_original(pos_bboxes[i].unsqueeze(0), clip_h, clip_w)[0]
                    y1, x1, y2, x2 = rec[0]
                    y1 = int(y1 if not isinstance(y1, torch.Tensor) else y1.item())
                    x1 = int(x1 if not isinstance(x1, torch.Tensor) else x1.item())
                    y2 = int(y2 if not isinstance(y2, torch.Tensor) else y2.item())
                    x2 = int(x2 if not isinstance(x2, torch.Tensor) else x2.item())

                    append_bbox(vid, frame, x1, y1, x2, y2)

        # flush any pending runs
        for vid in list(state.keys()):
            flush(vid)

    # serialize to list
    data = list(results.values())

    # ensure folder and write to output_path
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data