
import os
import io
import tempfile
import numpy as np
import pydicom
from fastapi import UploadFile

# Try importing CV2/TorchVision
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from torchvision.io import read_video
except ImportError:
    read_video = None

async def read_video_frames(uploaded_file: UploadFile) -> np.ndarray:
    """
    Returns frames as (T,H,W,3) uint8 for mp4/avi/mov/mkv and DICOM cine; supports .npy (pre-saved).
    """
    name = uploaded_file.filename.lower()
    content = await uploaded_file.read()
    
    if name.endswith(".npy"):
        try:
            return np.load(io.BytesIO(content))
        except Exception as e:
            raise RuntimeError(f"Failed to load npy: {e}")

    if name.endswith((".mp4", ".avi", ".mov", ".mkv")):
        # We need a temp file for video readers
        suf = os.path.splitext(name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            frames = []
            if read_video is not None:
                # torchvision
                vframes, _, _ = read_video(tmp_path, pts_unit="sec")
                frames = vframes.numpy()
            elif cv2 is not None:
                # fallback OpenCV
                cap = cv2.VideoCapture(tmp_path)
                ok, frame = cap.read()
                while ok:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    ok, frame = cap.read()
                cap.release()
                if not frames:
                     raise RuntimeError("OpenCV failed to decode video.")
                frames = np.stack(frames, axis=0).astype(np.uint8)
            else:
                raise RuntimeError("No video reader available (need torchvision or OpenCV).")
                
            return frames
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if name.endswith(".dcm"):
        try:
            ds = pydicom.dcmread(io.BytesIO(content))
        except Exception:
             raise RuntimeError("pydicom failed or not installed.")
             
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

def predict_coronary_dominance_from_rca_bytes(rca_bytes: bytes, suffix: str) -> str:
    try:
        # minimal read: average intensity
        if suffix.lower() == ".dcm":
            ds = pydicom.dcmread(io.BytesIO(rca_bytes))
            arr = ds.pixel_array.astype(np.float32)
            mean_int = arr.mean()
        else:
            if cv2 is None:
                return "Right Dominant" # Fallback
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(rca_bytes); tmp_path = tmp.name
                
            cap = cv2.VideoCapture(tmp_path)
            vals, ok = [], True
            while ok:
                ok, frame = cap.read()
                if ok:
                    vals.append(frame.mean())
            cap.release()
            os.remove(tmp_path)
            mean_int = float(np.mean(vals)) if vals else 0.0
            
        return "Right Dominant" if mean_int >= 80 else "Left Dominant"
    except Exception:
        return "Right Dominant"
