
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Optional

# Try importing CV2/Scikit-image for stenosis logic, strictly as in final_deploy.py
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from skimage.morphology import skeletonize as _sk_skeletonize
except ImportError:
    _sk_skeletonize = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

# ================== UNet GEN (PatchGAN generator) ==================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
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

def load_patchgan_generator(model_path: str, device: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Segmentation model not found: {model_path}")
    
    ckpt = torch.load(model_path, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", 2))
    gen = UNetGen(in_ch=1, out_ch=num_classes, base_c=32).to(device)
    
    if "gen_state_dict" not in ckpt:
        # Fallback or error if dict is different, but based on final_deploy.py this is expected
        raise KeyError("Checkpoint missing 'gen_state_dict'.")
        
    gen.load_state_dict(ckpt["gen_state_dict"])
    gen.eval()
    return gen, num_classes

@torch.no_grad()
def unet_infer_on_pil(pil_img: Image.Image, net: nn.Module, device: str) -> np.ndarray:
    """
    Input: PIL image (assumed grayscale coronary angio best), any size.
    Output: label_map (H,W) uint8 with values in {0..out_classes-1}
    """
    orig_w, orig_h = pil_img.size
    # Use grayscale channel (in_ch=1 model)
    gray = pil_img.convert("L")
    x = torch.from_numpy(np.array(gray)).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
    
    # Resize to network size (assume 256x256 training)
    # Note: final_deploy.py divides by 255.0 explicitly
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
    # clamp
    if n_classes > len(palette):
         n_classes = len(palette)
         
    mask_rgb = palette[np.clip(label_map, 0, len(palette) - 1)]
    
    if cv2 is not None:
        overlay = cv2.addWeighted(rgb, 0.7, mask_rgb, 0.3, 0)
    else:
        overlay = (0.7 * rgb + 0.3 * mask_rgb).astype(np.uint8)
    return overlay

# ================== Stenosis Finder ==================
def _thin_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    if _sk_skeletonize is not None:
        return _sk_skeletonize(binary_mask.astype(bool)).astype(np.uint8)
    # fallback: simple thinning code from final_deploy.py
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
