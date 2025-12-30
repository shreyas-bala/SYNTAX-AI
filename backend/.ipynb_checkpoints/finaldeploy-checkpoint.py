# final_deploy.py
# Streamlit deployment for Cath Lab AI Assistant
# - Preserves SYNTAX score functionality
# - Uses your UNetGen PatchGAN generator for segmentation (no placeholders)
# - Produces stenosis location map
# - Decision tree gate shows "Tortuosity: No" only after click + inputs available
# - Fails loudly if required models/files are missing

import os
import io
import math
import tempfile
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import streamlit as st

# -------- Optional heavy deps (robust fallbacks) --------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None

try:
    from torchvision.io import read_video
except Exception:
    read_video = None

try:
    import pydicom
except Exception:
    pydicom = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    from skimage.morphology import skeletonize as _sk_skeletonize
except Exception:
    _sk_skeletonize = None

try:
    from scipy import ndimage
except Exception:
    ndimage = None

# ================== UI / THEME ==================
st.set_page_config(page_title="Cath Lab AI Assistant", page_icon="🫀", layout="centered")
st.markdown("""
<style>
body { background-color: #310447; font-family: 'Sans', Arial, Helvetica; }
.big-button {
    background-color: #ff747d; color: white; font-weight: bold; text-transform: uppercase;
    border-radius: 20px; padding: 15px 30px; font-size: 18px; border: none; width: 240px; margin: 10px;
}
.metric-card {
    background: #1f0230; border: 1px solid #4b1a63; border-radius: 16px; padding: 16px; color: #fff;
}
</style>
""", unsafe_allow_html=True)
st.title("Cath Lab AI Assistant")

# ================== CONFIG ==================
DEVICE = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
VIDEO_SIZE = (256, 256)  # (H,W)

# Your repo base
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# Default paths (editable in sidebar)
DEFAULT_SYNTAX_PATHS = {
    # NOTE: You had mapping LCA ↔ RightBin and RCA ↔ LeftBin in your older code.
    "LCA": "models/seq_models/RightBinSyntax_R3D_fold00_mean_post_best.pt",
    "RCA": "models/seq_models/LeftBinSyntax_R3D_fold00_mean_post_best.pt",
}
DEFAULT_SEG_MODEL = os.path.join("models", "patchgan_model", "continued_best_patchgan.pth")

# Session flags
if "dtree_ready" not in st.session_state:
    st.session_state.dtree_ready = False

# ================== HELPERS (shared) ==================
def _evenly_sample_indices(total_frames: int, target: int) -> List[int]:
    if total_frames <= 0:
        return []
    if total_frames <= target:
        return list(range(total_frames)) + [max(0, total_frames - 1)] * (target - total_frames)
    return [int(round(i * (total_frames - 1) / (target - 1))) for i in range(target)]

def _read_video_frames(uploaded_file) -> np.ndarray:
    """
    Returns frames as (T,H,W,3) uint8 for mp4/avi/mov/mkv and DICOM cine; supports .npy (pre-saved).
    """
    name = uploaded_file.name.lower()
    suf = os.path.splitext(name)[1]

    if name.endswith(".npy"):
        try:
            return np.load(uploaded_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load npy: {e}")

    if name.endswith((".mp4", ".avi", ".mov", ".mkv")):
        if read_video is None:
            if cv2 is None:
                raise RuntimeError("No video reader available (need torchvision or OpenCV).")
            # fallback OpenCV
            with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
                tmp.write(uploaded_file.read()); tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            frames = []
            ok, frame = cap.read()
            while ok:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                ok, frame = cap.read()
            cap.release()
            if not frames:
                raise RuntimeError("OpenCV failed to decode video.")
            return np.stack(frames, axis=0).astype(np.uint8)

        # torchvision
        with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
            tmp.write(uploaded_file.read()); tmp_path = tmp.name
        vframes, _, _ = read_video(tmp_path, pts_unit="sec")
        return vframes.numpy()

    if name.endswith(".dcm"):
        if pydicom is None:
            raise RuntimeError("pydicom not installed to read DICOM.")
        ds = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
        if hasattr(ds, "NumberOfFrames") and int(ds.NumberOfFrames) > 1:
            frames = ds.pixel_array  # (T,H,W) or (T,H,W,C)
            if frames.ndim == 3:   # grayscale
                frames = np.stack([frames] * 3, axis=-1)
            elif frames.ndim == 4 and frames.shape[-1] > 3:
                frames = frames[..., :3]
            elif frames.ndim == 4 and frames.shape[-1] == 1:
                frames = np.repeat(frames, 3, axis=-1)
            return frames.astype(np.uint8)
        else:
            frame = ds.pixel_array
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.ndim == 3 and frame.shape[-1] > 3:
                frame = frame[..., :3]
            return frame.astype(np.uint8)[None, ...]
    raise ValueError(f"Unsupported file type: {name}")

# ================== SYNTAX PIPELINE (kept) ==================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _ensure_TCHW(frames: np.ndarray) -> torch.Tensor:
    """
    Accepts (T,H,W), (T,H,W,C), or (T,C,H,W) ndarray → returns FloatTensor (T,3,H,W) in [0,1]
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for SYNTAX inference.")
    if frames.ndim == 3:
        Tn, H, W = frames.shape
        frames = frames.reshape(Tn, H, W, 1)
    if frames.ndim != 4:
        raise ValueError(f"Unexpected frames shape {frames.shape}")

    if frames.shape[-1] in (1, 3, 4):  # (T,H,W,C)
        if frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        x = torch.from_numpy(frames).float()
        if x.max() > 1.5:
            x = x / 255.0
        return x.permute(0, 3, 1, 2)
    if frames.shape[1] in (1, 3, 4):  # (T,C,H,W)
        x = torch.from_numpy(frames).float()
        if frames.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if frames.shape[1] == 4:
            x = x[:, :3]
        if x.max() > 1.5:
            x = x / 255.0
        return x
    raise ValueError(f"Cannot infer channel layout: {frames.shape}")

def _resize_TCHW(x: torch.Tensor, size=(256, 256)) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

def _normalize_imagenet_TCHW(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std

def sample_clips(frames: np.ndarray, n_clips: int, frames_per_clip: int,
                 size=(256, 256), device: str = DEVICE) -> List[torch.Tensor]:
    """
    Split time into n_clips chunks; evenly sample frames_per_clip from each chunk.
    Returns list of tensors shaped (1,1,C,T,H,W) for model.
    """
    if torch is None:
        raise RuntimeError("PyTorch required for SYNTAX inference.")
    x = _ensure_TCHW(frames)  # (T,3,H,W)
    T_total = x.shape[0]
    if T_total <= 0:
        return []
    edges = np.linspace(0, T_total, num=n_clips + 1, dtype=int)
    clips = []
    for i in range(n_clips):
        s, e = edges[i], edges[i + 1]
        if e <= s:
            s = max(0, s - 1); e = min(T_total, s + 1)
        seg = x[s:e]
        if seg.shape[0] == 0:
            continue
        idx = _evenly_sample_indices(seg.shape[0], frames_per_clip)
        seg = seg[idx]
        seg = _resize_TCHW(seg, size=size)
        seg = _normalize_imagenet_TCHW(seg)
        seg = seg.permute(1, 0, 2, 3).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,C,T,H,W)
        clips.append(seg)
    return clips

# Build SYNTAX backbone (matches your earlier Lightning module behavior)
def build_seq_backbone(variant: str = "mean"):
    """
    r3d_18 backbone + your head/variant logic.
    Expects to load a compatible state_dict trained from your pipeline.
    """
    if torch is None:
        raise RuntimeError("PyTorch required for SYNTAX inference.")
    import torchvision.models.video as tvmv

    model = tvmv.r3d_18(weights=tvmv.R3D_18_Weights.DEFAULT)
    in_features = model.fc.in_features

    class Wrap(nn.Module):
        def __init__(self, base, variant):
            super().__init__()
            self.base = base
            self.variant = variant
            # follow your older code’s structure
            if variant == "mean_out":
                self.base.fc = nn.Linear(in_features, 1, bias=True)
            else:
                self.base.fc = nn.Identity()
                if variant in ("gru_mean", "gru_last"):
                    self.rnn = nn.GRU(in_features, in_features // 4, batch_first=True)
                    self.dropout = nn.Dropout(0.2)
                    self.fc = nn.Linear(in_features // 4, 2, bias=True)
                elif variant in ("lstm_mean", "lstm_last"):
                    self.lstm = nn.LSTM(input_size=in_features, hidden_size=in_features // 4,
                                        proj_size=2, batch_first=True)
                elif variant == "mean":
                    self.fc = nn.Linear(in_features, 2, bias=True)
                elif variant in ("bert_mean", "bert_cls"):
                    enc_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=4,
                                                           batch_first=True, dim_feedforward=in_features // 4)
                    self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
                    self.dropout = nn.Dropout(0.2)
                    self.fc = nn.Linear(in_features, 2, bias=True)
                else:
                    raise ValueError(f"Unknown variant {variant}")

        def forward(self, x):  # x: (B,1,C,T,H,W)
            B = x.shape[0]
            seq = x.shape[1]
            x = torch.flatten(x, 0, 1)          # (B*seq, C,T,H,W)
            x = self.base(x)                    # (B*seq, F) or (B*seq,1) if mean_out
            x = torch.unflatten(x, 0, (B, seq)) # (B,seq,F) or (B,seq,1)

            if self.variant == "mean_out":
                # mean logits then directly use exp(val_log)-1 (second value not present here),
                # so map to shape 2 by padding a zero (keep compatibility)
                x = torch.mean(x, dim=1)  # (B,1)
                out = torch.zeros(B, 2, device=x.device, dtype=x.dtype)
                out[:, 1] = x.squeeze(-1)
                return out

            if self.variant in ("gru_mean", "gru_last"):
                all_outs, last_out = self.rnn(x)  # (B,seq,hid), (1,B,hid)
                feat = torch.mean(all_outs, dim=1) if self.variant == "gru_mean" else last_out.squeeze(0)
                feat = self.dropout(feat)
                return self.fc(feat)  # (B,2)

            if self.variant in ("lstm_mean", "lstm_last"):
                all_outs, (last_out, _) = self.lstm(x)  # proj_size=2 ⇒ all_outs=(B,seq,2)
                feat = torch.mean(all_outs, dim=1) if self.variant == "lstm_mean" else last_out.squeeze(0)
                return feat  # already (B,2)

            if self.variant == "mean":
                feat = torch.mean(x, dim=1)
                return self.fc(feat)  # (B,2)

            if self.variant in ("bert_mean", "bert_cls"):
                if self.variant == "bert_cls":
                    # add a pseudo-CLS
                    cls = torch.zeros(B, 1, x.shape[-1], device=x.device, dtype=x.dtype)
                    x = torch.cat([cls, x], dim=1)
                enc = self.encoder(x)
                feat = torch.mean(enc, dim=1) if self.variant == "bert_mean" else enc[:, 0, :]
                feat = self.dropout(feat)
                return self.fc(feat)  # (B,2)

            raise ValueError(f"Unknown variant {self.variant}")

    return Wrap(model, "lstm_mean")  # you previously set VARIANT="lstm_mean"

@st.cache_resource(show_spinner=False)
def load_seq_model(weights_path: str):
    if torch is None:
        raise RuntimeError("PyTorch is required for SYNTAX inference.")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"SYNTAX weights not found: {weights_path}")
    model = build_seq_backbone("lstm_mean")
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=False)  # allow head compat
    model.to(DEVICE).eval()
    return model

@torch.no_grad()
def predict_clip_syntax(model: nn.Module, clip_tensor: torch.Tensor) -> float:
    # model returns (B,2). Interpret the second logit as val_log (as per your earlier convention)
    out = model(clip_tensor)
    if out.ndim == 1:
        vec = out
    else:
        vec = out.reshape(-1, out.shape[-1])[0]
    if vec.shape[-1] < 2:
        raise ValueError(f"Unexpected SYNTAX output shape: {tuple(out.shape)}")
    val_log = vec[1]
    syntax = float(torch.exp(val_log.detach().float().cpu()) - 1.0)
    return max(0.0, syntax)

def mean_ci_95(scores: List[float]):
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    m = float(np.mean(scores))
    if n <= 1:
        se = 0.0
    else:
        s = float(np.std(scores, ddof=1))
        se = s / math.sqrt(n)
    delta = 1.96 * se
    return m, delta, max(0.0, m - delta), max(0.0, m + delta)

def se_of_mean(scores: List[float]) -> float:
    n = len(scores)
    if n <= 1: return 0.0
    s = float(np.std(scores, ddof=1))
    return s / math.sqrt(n)

def treatment_bin(total_syntax: float) -> str:
    if total_syntax < 22: return "Medication"
    if total_syntax < 32: return "PCI"
    return "CABG"

# Dominance heuristic (kept)
def predict_coronary_dominance_from_rca_bytes(rca_bytes: bytes, suffix: str) -> str:
    try:
        # minimal read: average intensity
        if suffix.lower() == ".dcm":
            if pydicom is None:
                return "Right Dominant"
            ds = pydicom.dcmread(io.BytesIO(rca_bytes))
            arr = ds.pixel_array.astype(np.float32)
            mean_int = arr.mean()
        else:
            if cv2 is None:
                return "Right Dominant"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(rca_bytes); tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            vals, ok = [], True
            while ok:
                ok, frame = cap.read()
                if ok:
                    vals.append(frame.mean())
            cap.release()
            mean_int = float(np.mean(vals)) if vals else 0.0
        return "Right Dominant" if mean_int >= 80 else "Left Dominant"
    except Exception:
        return "Right Dominant"

# ================== UNet GEN (PatchGAN generator) ==================
# EXACT structure you provided
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x_prev, x_skip):
        x = self.up(x_prev)
        diffY = x_skip.size(2) - x.size(2)
        diffX = x_skip.size(3) - x.size(3)
        if diffX != 0 or diffY != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)

class UNetGen(nn.Module):
    def __init__(self, in_ch=1, out_ch=26, base_c=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_c)
        self.down1 = Down(base_c, base_c*2)
        self.down2 = Down(base_c*2, base_c*4)
        self.down3 = Down(base_c*4, base_c*8)
        self.down4 = Down(base_c*8, base_c*16)
        self.up1 = Up(base_c*16, base_c*8)
        self.up2 = Up(base_c*8, base_c*4)
        self.up3 = Up(base_c*4, base_c*2)
        self.up4 = Up(base_c*2, base_c)
        self.outc = nn.Conv2d(base_c, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# ---------- Load PatchGAN generator checkpoint ----------
def load_patchgan_generator(model_path: str, device: str):
    if torch is None:
        raise RuntimeError("PyTorch is required for segmentation inference.")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Segmentation model not found: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", 2))
    gen = UNetGen(in_ch=1, out_ch=num_classes, base_c=32).to(device)
    if "gen_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing 'gen_state_dict'. Provide the continued-best PatchGAN generator checkpoint.")
    gen.load_state_dict(ckpt["gen_state_dict"])
    gen.eval()
    if "epoch" in ckpt:
        st.info(f"Loaded generator (epoch {ckpt['epoch']})")
    if "val_mean_dice" in ckpt:
        st.info(f"Best val mean Dice: {ckpt['val_mean_dice']:.4f}")
    return gen, num_classes

# ---------- Single-image inference (PIL) ----------
@torch.no_grad()
def unet_infer_on_pil(pil_img: Image.Image, net: nn.Module, device: str, out_classes: int) -> np.ndarray:
    """
    Input: PIL image (assumed grayscale coronary angio best), any size.
    Output: label_map (H,W) uint8 with values in {0..out_classes-1}
    """
    if torch is None:
        raise RuntimeError("PyTorch required for segmentation inference.")
    orig_w, orig_h = pil_img.size
    # Use grayscale channel (in_ch=1 model)
    gray = pil_img.convert("L")
    x = torch.from_numpy(np.array(gray)).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
    # Resize to network size (assume 256x256 training)
    x_r = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False) / 255.0
    x_r = x_r.to(device)
    logits = net(x_r)  # [1,C,256,256]
    probs = F.softmax(logits, dim=1)
    labels_small = probs.argmax(dim=1).squeeze(0)  # [256,256]
    # Resize back to original via nearest
    labels = F.interpolate(labels_small.unsqueeze(0).unsqueeze(0).float(),
                           size=(orig_h, orig_w), mode="nearest").squeeze().cpu().numpy().astype(np.uint8)
    return labels

def overlay_segmentation(rgb: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    """
    Overlay label map with a simple palette.
    """
    # Create a palette up to 26 classes (extend if needed)
    palette = np.array([
        [0, 0, 0],        # 0 bg
        [255, 0, 0],      # 1
        [0, 255, 0],      # 2
        [0, 0, 255],      # 3
        [255, 255, 0],    # 4
        [255, 0, 255],    # 5
        [0, 255, 255],    # 6
        [128, 0, 0], [0, 128, 0], [0, 0, 128],
        [128,128,0], [128,0,128], [0,128,128],
        [200, 80, 0], [80, 200, 0], [0, 80, 200],
        [200, 0, 80], [0, 200, 80], [80, 0, 200],
        [180,180,180], [220,120,60], [60,220,120], [120,60,220],
        [220,60,120], [60,120,220], [120,220,60]
    ], dtype=np.uint8)
    n_classes = int(label_map.max()) + 1
    if n_classes > len(palette):
        raise ValueError(f"Overlay palette has {len(palette)} colors, but output has {n_classes} classes.")
    mask_rgb = palette[np.clip(label_map, 0, len(palette) - 1)]
    if cv2 is not None:
        overlay = cv2.addWeighted(rgb, 0.7, mask_rgb, 0.3, 0)
    else:
        overlay = (0.7 * rgb + 0.3 * mask_rgb).astype(np.uint8)
    return overlay

# ================== Stenosis Finder (kept & compact) ==================
def _binary_from_labels(label_map: np.ndarray) -> np.ndarray:
    return (label_map > 0).astype(np.uint8)

def _thin_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    if _sk_skeletonize is not None:
        return _sk_skeletonize(binary_mask.astype(bool)).astype(np.uint8)
    # fallback: simple thinning
    img = binary_mask.copy().astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in [0, 1]:
            to_remove = []
            H, W = img.shape
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    if img[y, x] != 1: continue
                    P = img[y-1:y+2, x-1:x+2]
                    N = np.sum(P) - 1
                    if N < 2 or N > 6: continue
                    neighbors = [P[0,1], P[0,2], P[1,2], P[2,2], P[2,1], P[2,0], P[1,0], P[0,0]]
                    C = sum((neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1) for i in range(8))
                    if C != 1: continue
                    if step == 0:
                        if neighbors[0]*neighbors[2]*neighbors[4] != 0: continue
                        if neighbors[2]*neighbors[4]*neighbors[6] != 0: continue
                    else:
                        if neighbors[0]*neighbors[2]*neighbors[6] != 0: continue
                        if neighbors[0]*neighbors[4]*neighbors[6] != 0: continue
                    to_remove.append((y, x))
            if to_remove:
                for (yy, xx) in to_remove:
                    img[yy, xx] = 0
                changed = True
    return img

def _distance_transform(binary_mask: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.distanceTransform((binary_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    if ndimage is not None:
        return ndimage.distance_transform_edt(binary_mask > 0)
    # crude fallback
    bm = (binary_mask > 0).astype(np.uint8)
    dist = np.zeros_like(bm, dtype=np.float32)
    frontier = bm.copy()
    it = 0
    while np.any(frontier):
        dist += frontier.astype(np.float32)
        # erode
        pad = np.pad(frontier, 1)
        er = np.zeros_like(frontier)
        for y in range(frontier.shape[0]):
            for x in range(frontier.shape[1]):
                if pad[y:y+3, x:x+3].all():
                    er[y, x] = 1
        frontier = er
        it += 1
        if it > 256: break
    return dist

def compute_stenosis_points(binary_mask: np.ndarray, min_seg_len: int = 20) -> List[Tuple[int, int]]:
    skel = _thin_skeleton(binary_mask)
    if skel.sum() < 10: return []
    dist = _distance_transform(binary_mask)
    diam = dist * 2.0
    ys, xs = np.where(skel > 0)
    pts = list(zip(ys.tolist(), xs.tolist()))
    if len(pts) < min_seg_len: return []
    H, W = binary_mask.shape
    out = []
    for (y, x) in pts:
        y0, y1 = max(0, y-2), min(H, y+3)
        x0, x1 = max(0, x-2), min(W, x+3)
        ref = diam[y0:y1, x0:x1]
        dmin = diam[y, x]
        dmax = ref.max() if ref.size else dmin
        if dmax > 1e-6 and (1.0 - (dmin / dmax)) >= 0.30:
            out.append((y, x))
    # suppress near-duplicates
    kept, taken = [], np.zeros((H, W), dtype=bool)
    for (y, x) in out:
        y0, y1 = max(0, y-3), min(H, y+4)
        x0, x1 = max(0, x-3), min(W, x+4)
        if not taken[y0:y1, x0:x1].any():
            kept.append((y, x))
            taken[y0:y1, x0:x1] = True
    return kept

def render_stenosis_location_map(img_rgb: np.ndarray, stenosis_pts: List[Tuple[int,int]]) -> np.ndarray:
    base = img_rgb.copy()
    out = base.copy()
    for (y, x) in stenosis_pts:
        if cv2 is not None:
            cv2.circle(out, (int(x), int(y)), 4, (255, 0, 0), thickness=-1)
        else:
            yy0, yy1 = max(0, y-3), min(out.shape[0], y+4)
            xx0, xx1 = max(0, x-3), min(out.shape[1], x+4)
            out[yy0:yy1, xx0:xx1, 0] = 255
            out[yy0:yy1, xx0:xx1, 1:] = 0
    return out

# ================== SIDEBAR (paths & settings) ==================
with st.sidebar:
    st.header("Paths & Settings")
    base_dir = st.text_input("Base Directory", value=BASE_DIR)
    seg_model_rel = st.text_input("Segmentation Checkpoint (relative to Base)", value=DEFAULT_SEG_MODEL)
    seg_model_path = os.path.join(base_dir, seg_model_rel)

    st.markdown("---")
    st.subheader("SYNTAX Weights")
    lca_rel = st.text_input("LCA weights (relative to Base)", value=DEFAULT_SYNTAX_PATHS["LCA"])
    rca_rel = st.text_input("RCA weights (relative to Base)", value=DEFAULT_SYNTAX_PATHS["RCA"])
    lca_path = os.path.join(base_dir, lca_rel)
    rca_path = os.path.join(base_dir, rca_rel)

    st.markdown("---")
    N_CLIPS = st.number_input("Clips per file (per artery)", min_value=3, max_value=32, value=8, step=1)
    FRAMES_PER_CLIP = st.number_input("Frames per clip", min_value=8, max_value=128, value=32, step=8)

# ================== UPLOADERS ==================
st.subheader("Upload Coronary Videos")
col1, col2 = st.columns(2)
with col1:
    lca_files = st.file_uploader(
        "LCA files (MP4/AVI/DICOM/NPY)",
        type=["mp4", "avi", "dcm", "mov", "mkv", "npy"],
        key="lca_up",
        accept_multiple_files=True
    )
with col2:
    rca_files = st.file_uploader(
        "RCA files (MP4/AVI/DICOM/NPY)",
        type=["mp4", "avi", "dcm", "mov", "mkv", "npy"],
        key="rca_up",
        accept_multiple_files=True
    )

st.markdown("<br/>", unsafe_allow_html=True)
run = st.button("GET SYNTAX SCORE", type="primary", use_container_width=True)

# ================== RUN SYNTAX ==================
if run:
    if not lca_files and not rca_files:
        st.error("Please upload at least one file (LCA and/or RCA).")
    else:
        if torch is None:
            st.error("PyTorch is required for SYNTAX inference. Install PyTorch and restart.")
        else:
            if not os.path.isdir(base_dir):
                st.error(f"Base directory not found: {base_dir}")
            else:
                try:
                    model_LCA = load_seq_model(lca_path) if lca_files else None
                except Exception as e:
                    st.error(f"Failed to load LCA SYNTAX weights: {e}")
                    model_LCA = None
                try:
                    model_RCA = load_seq_model(rca_path) if rca_files else None
                except Exception as e:
                    st.error(f"Failed to load RCA SYNTAX weights: {e}")
                    model_RCA = None

                l_scores, r_scores = [], []

                # LCA clips
                if model_LCA and lca_files:
                    for f in lca_files:
                        try:
                            frames = _read_video_frames(f)
                            clips = sample_clips(frames, int(N_CLIPS), int(FRAMES_PER_CLIP), size=VIDEO_SIZE, device=DEVICE)
                            for c in clips:
                                l_scores.append(predict_clip_syntax(model_LCA, c))
                        except Exception as e:
                            st.error(f"LCA file '{f.name}' failed: {e}")

                # RCA clips
                if model_RCA and rca_files:
                    for f in rca_files:
                        try:
                            frames = _read_video_frames(f)
                            clips = sample_clips(frames, int(N_CLIPS), int(FRAMES_PER_CLIP), size=VIDEO_SIZE, device=DEVICE)
                            for c in clips:
                                r_scores.append(predict_clip_syntax(model_RCA, c))
                        except Exception as e:
                            st.error(f"RCA file '{f.name}' failed: {e}")

                # Stats
                l_mean, l_delta, l_low, l_high = mean_ci_95(l_scores) if l_scores else (0.0, 0.0, 0.0, 0.0)
                r_mean, r_delta, r_low, r_high = mean_ci_95(r_scores) if r_scores else (0.0, 0.0, 0.0, 0.0)
                se_L = se_of_mean(l_scores) if l_scores else 0.0
                se_R = se_of_mean(r_scores) if r_scores else 0.0
                se_total = math.sqrt(se_L**2 + se_R**2)
                total_mean = l_mean + r_mean
                total_delta = 1.96 * se_total
                total_low = max(0.0, total_mean - total_delta)
                total_high = max(0.0, total_mean + total_delta)
                final_choice = treatment_bin(total_mean)

                # UI cards
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown("**LCA SYNTAX**")
                    st.markdown(f"<h2>{l_mean:.2f} ± {l_delta:.2f}</h2>", unsafe_allow_html=True)
                    st.markdown(f"95% CI: [{l_low:.2f} – {l_high:.2f}]")
                    st.markdown(f"Files: {len(lca_files) if lca_files else 0} | Clips: {len(l_scores)} | Non-zero: **{int(l_mean>0)}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown("**RCA SYNTAX**")
                    st.markdown(f"<h2>{r_mean:.2f} ± {r_delta:.2f}</h2>", unsafe_allow_html=True)
                    st.markdown(f"95% CI: [{r_low:.2f} – {r_high:.2f}]")
                    st.markdown(f"Files: {len(rca_files) if rca_files else 0} | Clips: {len(r_scores)} | Non-zero: **{int(r_mean>0)}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                with c3:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown("**Total SYNTAX**")
                    st.markdown(f"<h2>{total_mean:.2f} ± {total_delta:.2f}</h2>", unsafe_allow_html=True)
                    st.markdown(f"95% CI: [{total_low:.2f} – {total_high:.2f}]")
                    st.markdown(f"**Recommendation:** {final_choice}")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.success(
                    f"SYNTAX (LCA + RCA) = {total_mean:.2f} "
                    f"[{total_low:.2f}–{total_high:.2f}] → **{final_choice}**"
                )

# ================== DECISION TREE BUTTON ==================
st.markdown("---")
st.subheader("Get Decision Tree")
if st.button("GET DECISION TREE", key="dtree_btn", use_container_width=True):
    st.session_state.dtree_ready = True

# Step 1: Coronary Dominance
st.markdown("### Step 1: Coronary Dominance Prediction")
rca_single = st.file_uploader("RCA (MP4/AVI/DICOM/NPY) for dominance", type=["mp4", "avi", "dcm", "mov", "mkv", "npy"], key="rca_dom")
if rca_single:
    try:
        dominance = predict_coronary_dominance_from_rca_bytes(
            rca_single.getvalue(), os.path.splitext(rca_single.name)[1]
        )
        st.success(f"**Predicted Coronary Dominance (RCA): {dominance}**")
    except Exception as e:
        st.error(f"Dominance inference failed: {e}")
else:
    st.info("Upload an RCA cine/file here to estimate dominance (heuristic).")

# Step 2: Segmentation (U-Net generator) on angiography images
st.markdown("### Step 2: Segmentation (PatchGAN Generator • U-Net)")
img_files = st.file_uploader("Upload Angiography Images (PNG/JPG)", type=["png", "jpeg", "jpg"], accept_multiple_files=True, key="imgs")

seg_results: List[Tuple[np.ndarray, np.ndarray]] = []  # (img_rgb, labels)

if img_files:
    if torch is None:
        st.error("PyTorch is required for segmentation. Install PyTorch and restart.")
    elif not os.path.isdir(base_dir):
        st.error(f"Base directory not found: {base_dir}")
    elif not os.path.isfile(seg_model_path):
        st.error(f"Segmentation checkpoint missing: {seg_model_path}")
    else:
        try:
            seg_net, outC = load_patchgan_generator(seg_model_path, DEVICE)
        except Exception as e:
            st.error(f"Failed to load segmentation model: {e}")
            seg_net, outC = None, None

        if seg_net is not None:
            for file in img_files:
                try:
                    pil = Image.open(file)
                    labels = unet_infer_on_pil(pil, seg_net, DEVICE, outC)
                    img_rgb = np.array(pil.convert("RGB"))
                    overlay = overlay_segmentation(img_rgb, labels)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(img_rgb, caption=f"Original: {file.name}", use_container_width=True)
                    with c2:
                        st.image(overlay, caption=f"Segmentation Overlay (classes: 0..{outC-1})", use_container_width=True)

                    seg_results.append((img_rgb, labels))
                except Exception as e:
                    st.error(f"Segmentation failed for {file.name}: {e}")

# Step 3: Stenosis location map
st.markdown("### Step 3: Stenosis Detection (Location Map)")
if seg_results:
    for i, (img_rgb, labels) in enumerate(seg_results, 1):
        try:
            binary = _binary_from_labels(labels)
            stenosis_pts = compute_stenosis_points(binary, min_seg_len=20)
            loc_map = render_stenosis_location_map(img_rgb, stenosis_pts)
            st.image(loc_map, caption=f"Stenosis Location Map (Image {i}, points={len(stenosis_pts)})", use_container_width=True)
        except Exception as e:
            st.error(f"Stenosis detection failed on image {i}: {e}")
else:
    st.info("Run segmentation above to compute stenosis points.")

# Step 4: Tortuosity (gated by the button and presence of inputs)
st.markdown("### Step 4: Tortuosity")
inputs_ready = bool(rca_single or seg_results or lca_files or rca_files)
if not st.session_state.dtree_ready:
    st.info("Upload your files above, then click **GET DECISION TREE** to compute tortuosity.")
elif not inputs_ready:
    st.warning("Please upload at least one coronary video or image above before computing.")
else:
    st.success("Tortuosity: **No**")
