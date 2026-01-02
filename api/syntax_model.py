# syntax_model.py
# Core SYNTAX inference logic extracted from Streamlit app
# Used by FastAPI backend

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# ---------------- CONFIG ----------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------- FRAME UTILS ----------------
def _evenly_sample_indices(total_frames: int, target: int) -> List[int]:
    if total_frames <= target:
        return list(range(total_frames)) + [total_frames - 1] * (target - total_frames)
    return [int(round(i * (total_frames - 1) / (target - 1))) for i in range(target)]

def _ensure_TCHW(frames: np.ndarray) -> torch.Tensor:
    if frames.ndim == 3:
        frames = frames[..., None]
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    x = torch.from_numpy(frames).float()
    if x.max() > 1.5:
        x /= 255.0
    return x.permute(0, 3, 1, 2)  # (T,3,H,W)

def _normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(1,3,1,1).to(x.device)
    std  = torch.tensor(IMAGENET_STD).view(1,3,1,1).to(x.device)
    return (x - mean) / std

# ---------------- MODEL ----------------
def build_seq_backbone():
    import torchvision.models.video as tvm

    base = tvm.r3d_18(weights=tvm.R3D_18_Weights.DEFAULT)
    in_f = base.fc.in_features
    base.fc = nn.Identity()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = base
            self.lstm = nn.LSTM(in_f, in_f // 4, proj_size=2, batch_first=True)

        def forward(self, x):
            B, S, C, T, H, W = x.shape
            x = x.view(B * S, C, T, H, W)
            feats = self.backbone(x)
            feats = feats.view(B, S, -1)
            out, _ = self.lstm(feats)
            return out.mean(dim=1)

    return Model()

def load_seq_model(path: str, device: str):
    model = build_seq_backbone()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model

# ---------------- INFERENCE ----------------
@torch.no_grad()
def run_syntax_inference(
    video: np.ndarray,
    model: nn.Module,
    artery: str,
    n_clips: int = 4,
    frames_per_clip: int = 16,
    device: str = "cpu"
) -> float:
    device = next(model.parameters()).device
    x = _ensure_TCHW(video)
    T = x.shape[0]

    clips = []
    edges = np.linspace(0, T, n_clips + 1, dtype=int)

    for i in range(n_clips):
        seg = x[edges[i]:edges[i+1]]
        if len(seg) == 0:
            continue
        idx = _evenly_sample_indices(len(seg), frames_per_clip)
        clip = seg[idx]
        clip = F.interpolate(clip, size=(256,256), mode="bilinear")
        clip = _normalize(clip)
        clip = clip.permute(1,0,2,3).unsqueeze(0).unsqueeze(0).to(device)
        clips.append(clip)

    scores = []
    for c in clips:
        out = model(c)[0,1]
        score = torch.exp(out).item() - 1
        score = max(0.0, score)   # 🔥 clamp
        scores.append(score)


    if not scores:
        return 0.0

    return round(float(np.mean(scores)), 2)
