# AeroEyes Demo

Visual object tracking in drone videos using deep learning.

## Requirements

- Python 3.10+
- CUDA GPU (recommended)
- FFmpeg

## Installation

```bash
pip install -r requirements.txt
pip install gradio
```

## Download Model

Download checkpoint from Google Drive:
```bash
pip install gdown
gdown https://drive.google.com/uc?id=1gNb99zuOe4Gjpd4yabZZGnJCBSd-m0ED -O cpt_best_iou.pth.tar
```

Or download manually: https://drive.google.com/file/d/1gNb99zuOe4Gjpd4yabZZGnJCBSd-m0ED

## Run Demo

```bash
python demo/app.py
```

Open browser: http://localhost:7860

## Docker

Build:
```bash
docker build -t aeroeyes .
```

Run:
```bash
docker run --gpus all -p 7860:7860 -v $(pwd)/cpt_best_iou.pth.tar:/app/cpt_best_iou.pth.tar aeroeyes
```

## Usage

1. Upload drone video (MP4)
2. Upload query images of target object
3. Set detection threshold (default: 0.5)
4. Click "Process Video"
5. Download output video with bounding boxes

## Input/Output

| Input | Format |
|-------|--------|
| Video | MP4, AVI |
| Query Images | JPG, PNG |

| Output | Description |
|--------|-------------|
| Video | MP4 with green bounding boxes |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Threshold | 0.5 | Detection confidence (0.1-0.9) |

## Project Structure

```
AeroEyes/
├── demo/
│   └── app.py          # Gradio demo
├── model/              # Model architecture
├── config/             # Configuration files
├── cpt_best_iou.pth.tar  # Model checkpoint
└── requirements.txt
```
