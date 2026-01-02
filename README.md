# SYNTAX-AI 
**AI-assisted SYNTAX Score Estimation and Explainable Coronary Decision Support**

SYNTAX-AI is an end-to-end AI system that automatically estimates the **SYNTAX score** from coronary angiography and provides **explainable decision support** to aid cardiologists in choosing between **medical management, PCI, or CABG**.

The platform combines:
- Deep learning–based video analysis
- Vessel segmentation and stenosis localization
- Statistical aggregation with confidence intervals
- A transparent, stepwise decision tree

---

##  Clinical Motivation

The SYNTAX score is routinely used to guide revascularization strategy, but:
- It is **manually computed**
- Requires answering **~11 structured questions**
- Is **time-consuming and inter-observer dependent**

SYNTAX-AI automates this process while **preserving interpretability**, enabling faster and more consistent clinical decisions.

---

##  System Overview

### 1. SYNTAX Score Estimation (Video-based)
- Input: Coronary angiography videos (`.mp4`, `.avi`, `.dcm`, `.npy`)
- Backbone: **3D ResNet (R3D-18)**
- Temporal Modeling: **LSTM-based aggregation**
- Separate inference for:
  - **LCA**
  - **RCA**
- Output:
  - Mean SYNTAX score
  - 95% Confidence Interval
  - Treatment recommendation

**Treatment bins:**
- `< 22` → Medical Therapy  
- `22–32` → PCI  
- `> 32` → CABG  

---

### 2. Explainability Pipeline (Decision Tree)

SYNTAX-AI provides visual and algorithmic explainability via:

#### a) Coronary Dominance
- Heuristic dominance estimation from RCA cine

#### b) Vessel Segmentation
- Model: **PatchGAN-trained U-Net Generator**
- Output: Multi-class vessel segmentation map

#### c) Stenosis Localization
- Skeletonization + distance transform
- Identifies focal diameter reductions ≥30%
- Produces **stenosis location maps**

#### d) Tortuosity (Demo-gated)
- Shown as part of structured decision flow

---

##  User Interface

The system is deployed using **Streamlit** and supports:
- Multi-file uploads (LCA + RCA)
- Real-time inference on GPU
- Visual overlays and confidence reporting
- Stepwise decision-tree exploration

---

##  Repository Structure

syntax-ai/
│
├── backend/
│ └── final_deploy.py # Streamlit application (main entry point)
│
├── models/
│ ├── seq_models/
│ │ ├── RightBinSyntax_R3D_fold00_mean_post_best.pt
│ │ └── LeftBinSyntax_R3D_fold00_mean_post_best.pt
│ │
│ └── patchgan_model/
│ └── continued_best_patchgan.pth
│
├── .gitattributes # Git LFS configuration
├── README.md

yaml
Copy code

>  Model weights are stored using **Git LFS**.

---

##  Installation

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA-enabled GPU (recommended)

### Setup

git clone https://github.com/shreyas-bala/SYNTAX-AI.git
cd SYNTAX-AI

conda create -n syntax-ai python=3.9
conda activate syntax-ai

pip install -r requirements.txt
Running the Application

cd backend
streamlit run final_deploy.py
The app will launch locally and can utilize GPU acceleration if available.


# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template]

 Disclaimer
This project is for research and demonstration purposes only.
It is not a certified medical device and must not be used for clinical decision-making without regulatory approval.

 Author
Shreyas Balakarthikeyan
IIT Madras
Medical Sciences & Engineering
(https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
