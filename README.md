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

---
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
