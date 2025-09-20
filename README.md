# Intel Image Classification - Task 1  

This project demonstrates **basic image fundamentals and preprocessing** using the **Intel Image Classification Dataset**.  
It is divided into three main scripts, each focusing on different aspects of image handling and processing.  

---

## 📂 Files Overview

### 1. `intel_image_classification_project.py`
- Loads dataset and checks folder structure  
- Counts images per class (train/test/predict)  
- Displays a grid of sample images from the dataset  

### 2. `intel_image_task1_image_fundamentals.py`
- Performs basic image operations:  
  - Reading and displaying images  
  - Exploring color spaces (RGB, Gray, HSV, LAB)  
  - Resizing images  
  - Applying Gaussian Blur  
  - Edge detection (Canny)  

### 3. `step3_preprocess.py`
- Preprocesses images for training  
- Operations include:  
  - Resizing to 224×224  
  - Converting to Grayscale  
  - Applying Gaussian Blur  
  - Edge Detection (Canny / skimage / PIL fallback)  
- Saves processed images into a new folder (`processed_v1`)  

# Intel Image Classification - Task 1 (Image Fundamentals)

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# ========================
# Configurations
# ========================
BASE = Path(r"\\DESKTOP-16ABCU6\Users\SRComputers\Downloads\intel_images")
SPLITS = {
    "seg_train": BASE / "seg_train" / "seg_train",
    "seg_test":  BASE / "seg_test" / "seg_test",
    "seg_pred":  BASE / "seg_pred" / "seg_pred",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
random.seed(42)

def imshow_rgb(img_rgb, title=""):
    plt.figure(figsize=(5,4))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

# ========================
# 1) Classes & sanity check
# ========================
for split, split_dir in SPLITS.items():
    if not split_dir.exists():
        print(f"[WARN] Missing split: {split}")
        continue
    classes = [p.name for p in split_dir.iterdir() if p.is_dir()]
    print(f"{split}: {len(classes)} classes -> {classes}")

# ========================
# 2) Image counts per class
# ========================
counts = defaultdict(int)
for split, split_dir in SPLITS.items():
    if not split_dir.exists():
        continue
    for cls_dir in split_dir.iterdir():
        if cls_dir.is_dir():
            n = sum(1 for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
            counts[f"{split}/{cls_dir.name}"] = n
            print(f"{split}/{cls_dir.name}: {n} images")

# ========================
# 3) Sample grid preview (train set)
# ========================
train_dir = SPLITS["seg_train"]
sample_paths = list(train_dir.rglob("*"))
sample_paths = [p for p in sample_paths if p.suffix.lower() in IMG_EXTS]
print("Total train images:", len(sample_paths))

if sample_paths:
    plt.figure(figsize=(12,6))
    for i, p in enumerate(sample_paths[:6]):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None: 
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.subplot(2,3,i+1)
        plt.imshow(img_rgb)
        plt.title(p.parent.name)
        plt.axis("off")
    plt.show()
else:
    print("❌ No training images found. Please check dataset path.")

# ========================
# 4) Image fundamentals on one sample
# ========================
if sample_paths:
    demo = sample_paths[0]
    img_bgr = cv2.imread(str(demo))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 4.1 Properties
    h, w, c = img_rgb.shape
    print(f"Properties -> height:{h}, width:{w}, channels:{c}")

    # 4.2 Color spaces
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    imshow_rgb(img_rgb, "Original (RGB)")
    plt.figure(figsize=(12,3))
    for i,(im,title) in enumerate([
        (gray,"Grayscale"), 
        (cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB),"HSV→RGB (viz)"),
        (cv2.cvtColor(lab, cv2.COLOR_LAB2RGB),"LAB→RGB (viz)")
    ]):
        plt.subplot(1,3,i+1)
        cmap = "gray" if title=="Grayscale" else None
        plt.imshow(im, cmap=cmap)
        plt.title(title)
        plt.axis("off")
    plt.show()

    # 4.3 Basic operations
    resized = cv2.resize(img_bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    blur = cv2.GaussianBlur(gray, (5,5), sigmaX=1.0)
    edges = cv2.Canny(blur, threshold1=100, threshold2=200)

    imshow_rgb(resized_rgb, "Resized (0.5x)")
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(blur, cmap="gray"); plt.title("Gaussian Blur"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(edges, cmap="gray"); plt.title("Canny Edges"); plt.axis("off")
    plt.show()
else:
    print("⚠️ Skipping Step 4: No training images to demo.")

