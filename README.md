# Ultrasafe 🔬

Real-time ultrasound nerve detection system using U-Net deep learning architecture, with a live streaming interface via OBS Virtual Camera.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-green)
![React](https://img.shields.io/badge/React-TypeScript-blue)
![License](https://img.shields.io/badge/License-AGPL--3.0-lightgrey)

## Overview
Ultrasafe is an end-to-end deep learning application for real-time ultrasound nerve segmentation. It combines a lightweight U-Net model trained on ultrasound images with a FastAPI backend and React frontend, enabling live segmentation overlay visualization through OBS Virtual Camera streaming.

## Demo
<img width="904" height="372" alt="image" src="https://github.com/user-attachments/assets/634e030e-4536-4213-a8b0-1a7fefc451b4" />
<div align="left">
  <img width="30%" height="200" alt="image" src="https://github.com/user-attachments/assets/e070ce9c-7da4-4bc6-b277-ae4441855e09" />
  <img width="30%" height="200" alt="image" src="https://github.com/user-attachments/assets/da7d9b38-5f5c-49cb-8663-5b2d913266d4" />
  <img width="30%" height="200" alt="image" src="https://github.com/user-attachments/assets/87a5bbd4-1a45-47f8-8c03-6c6870866af6" />
</div>

## Presentation
[📊 View Project Slides](https://docs.google.com/presentation/d/1uBQou69hAN5NAvLszqUvG3JG5SUJyMCTmnT1ZPLe7AA/edit?usp=sharing)

## Tech Stack
- **Deep Learning:** TensorFlow/Keras (U-Net architecture)
- **Backend:** FastAPI + Python
- **Frontend:** React + TypeScript + Vite
- **Streaming:** OBS Virtual Camera via MJPEG
- **Training:** Jupyter Notebooks

## Project layout
- `frontend/` - UI app (React/TS + Vite)
- `ultrasafe/` - backend API (FastAPI + PyTorch)
- `notebooks/` - Jupyter notebooks for data exploration and training
- `raw_data/` - original datasets (do not modify)
- `ultrasafe/data/` - curated data artifacts for training/inference
- `tests/` - tests

## Quick Start

### 1. Install backend dependencies
```bash
# Mac/Linux
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Build the frontend
```bash
cd frontend
npm install
npm run build
```

### 3. Run the backend
```bash
python -m ultrasafe
```

Open `http://localhost:8000`

## Frontend Dev Mode (optional)
Use this only if you want hot-reload during UI development:
```bash
cd frontend
npm install
set VITE_API_BASE_URL=http://localhost:8000
npm run dev
```
Open `http://localhost:5173`

## Training
The full training pipeline is available in the notebooks folder:
- `notebooks/unet_train_results.ipynb` — U-Net training with results, loss curves, Dice scores, and prediction visuals

## Live OBS Virtual Camera Streaming
The backend exposes a live MJPEG stream with segmentation overlay at:
`/video/overlay`

| Variable | Default | Description |
|---|---|---|
| `OBS_CAMERA_INDEX` | `1` | OBS camera index |
| `USE_FULL_FRAME` | `0` | Use full frame (`1`) or ROI (`0`) |
| `ROI_X`, `ROI_Y`, `ROI_W`, `ROI_H` | - | Region of interest coordinates |
| `OVERLAY_ALPHA` | `0.35` | Segmentation overlay transparency |

## Configuration
| Variable | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Input camera index |
| `MODEL_PATH` | - | Path to trained `.keras` model |
| `INFER_SIZE` | `256` | Inference image size |
| `USE_CUDA` | `0` | Enable GPU (`1`) or force CPU (`0`) |

## Built With
- [Thomas P.](https://github.com/thomas-code-lab)
- [Stacey](https://github.com/staceylqy/Ultrasafe)
- [Hugo](https://github.com/learnhugo)
