
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
import numpy as np
import torch
import os
import io
import base64
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(BASE_DIR)

# Import models & utils
from syntax_model import load_seq_model, run_syntax_inference
from segmentation_model import load_patchgan_generator, unet_infer_on_pil, overlay_segmentation, compute_stenosis_points, render_stenosis_location_map
from utils import read_video_frames, predict_coronary_dominance_from_rca_bytes

# Paths
LCA_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "seq_models", "RightBinSyntax_R3D_fold00_mean_post_best.pt")
RCA_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "seq_models", "LeftBinSyntax_R3D_fold00_mean_post_best.pt")
SEG_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "patchgan_model", "continued_best_patchgan.pth")

app = FastAPI(title="SYNTAX-AI Backend")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
model_LCA = None
model_RCA = None
model_SEG = None
seg_classes = None

@app.on_event("startup")
def load_models():
    global model_LCA, model_RCA, model_SEG, seg_classes
    
    # Load SYNTAX models
    try:
        if os.path.exists(LCA_MODEL_PATH):
            model_LCA = load_seq_model(LCA_MODEL_PATH, DEVICE)
            print("LCA model loaded")
        else:
            print(f"Warning: LCA model not found at {LCA_MODEL_PATH}")
            
        if os.path.exists(RCA_MODEL_PATH):
            model_RCA = load_seq_model(RCA_MODEL_PATH, DEVICE)
            print("RCA model loaded")
        else:
            print(f"Warning: RCA model not found at {RCA_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading SYNTAX models: {e}")

    # Load Segmentation model
    try:
        if os.path.exists(SEG_MODEL_PATH):
            model_SEG, seg_classes = load_patchgan_generator(SEG_MODEL_PATH, DEVICE)
            print("Segmentation model loaded")
        else:
             print(f"Warning: Segmentation model not found at {SEG_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading Segmentation model: {e}")


@app.get("/")
def root():
    return {"status": "SYNTAX-AI backend running", "device": DEVICE}

@app.post("/predict/syntax")
async def predict_syntax(
    lca: Optional[UploadFile] = File(None),
    rca: Optional[UploadFile] = File(None)
):
    if not lca and not rca:
        return {"error": "At least one of LCA or RCA must be provided"}

    results = {}

    if lca and model_LCA:
        try:
            lca_np = await read_video_frames(lca)
            results["lca_syntax"] = run_syntax_inference(lca_np, model_LCA, artery="LCA", device=DEVICE)
        except Exception as e:
            results["lca_error"] = str(e)

    if rca and model_RCA:
        try:
            rca_np = await read_video_frames(rca)
            results["rca_syntax"] = run_syntax_inference(rca_np, model_RCA, artery="RCA", device=DEVICE)
        except Exception as e:
             results["rca_error"] = str(e)

    lca_val = max(0.0, results.get("lca_syntax", 0))
    rca_val = max(0.0, results.get("rca_syntax", 0))
    total = lca_val + rca_val

    if total < 22:
        decision = "Medication"
    elif total < 32:
        decision = "PCI"
    else:
        decision = "CABG"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "syntax": results,
        "total_syntax": round(total, 2),
        "recommendation": decision
    }

@app.post("/predict/dominance")
async def predict_dominance(file: UploadFile = File(...)):
    """
    Predict coronary dominance from RCA video/image.
    """
    try:
        content = await file.read()
        suffix = os.path.splitext(file.filename)[1]
        dominance = predict_coronary_dominance_from_rca_bytes(content, suffix)
        return {"dominance": dominance}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict/segmentation")
async def predict_segmentation(file: UploadFile = File(...)):
    """
    Perform vessel segmentation and stenosis localization on an image.
    Returns base64 encoded images.
    """
    if not model_SEG:
        return JSONResponse(status_code=503, content={"error": "Segmentation model not loaded"})

    try:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        
        # Inference
        labels = unet_infer_on_pil(pil_img, model_SEG, DEVICE)
        
        # Overlay
        img_rgb = np.array(pil_img)
        overlay_rgb = overlay_segmentation(img_rgb, labels)
        
        # Stenosis
        binary_mask = (labels > 0).astype(np.uint8)
        stenosis_pts = compute_stenosis_points(binary_mask)
        stenosis_map_rgb = render_stenosis_location_map(img_rgb, stenosis_pts)
        
        # Convert to Base64
        def to_b64(img_arr):
            pil = Image.fromarray(img_arr)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "overlay_b64": to_b64(overlay_rgb),
            "stenosis_map_b64": to_b64(stenosis_map_rgb),
            "stenosis_count": len(stenosis_pts),
            "stenosis_locations": stenosis_pts
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
