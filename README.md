# MAM80731: Quality-Aware CNN for Glaucoma Detection

## Overview
This repository contains the implementation of a Convolutional Neural Network (CNN) 
for glaucoma detection using the Hillel Yaffe Glaucoma Dataset (HYGD). 
The project compares a baseline CNN with a quality-aware CNN that leverages 
image quality scores during training.

## Dataset
- **Source:** Hillel Yaffe Glaucoma Dataset (HYGD)
- **Link:** https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/
- **Images:** 747 retinal fundus images
- **Patients:** 288 patients
- **Labels:** GON+ (548 images) and GON- (199 images)

> **Note:** You must register at PhysioNet to download the dataset.

## Repository Structure
MAM80731-Glaucoma-CNN/
-glaucoma_cnn.ipynb   # Main notebook (EDA, training, evaluation)
-README.md

## Requirements
All code runs on **Google Colab** (recommended). No local installation needed.

Required libraries (all available in Colab):
- torch
- torchvision
- pandas
- numpy
- matplotlib
- scikit-learn
- Pillow
- grad-cam

## How to Reproduce

### 1. Open in Google Colab
Click the badge or open [Google Colab](https://colab.research.google.com/) 
and upload `glaucoma_cnn.ipynb`.

### 2. Mount Google Drive
Run the first cell to mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Prepare Dataset
Download HYGD from PhysioNet and upload to Google Drive with this structure:
MyDrive/glaucoma_project/Labels.csv/images/(all 747 .jpg files)

### 4. Update Path
In the notebook, update this variable to match your Drive path:
```python
project_dir = '/content/drive/MyDrive/glaucoma_project'
image_dir   = '/content/drive/MyDrive/glaucoma_project/images'
csv_path    = '/content/drive/MyDrive/glaucoma_project/Labels.csv'
```

### 5. Run All Cells
Go to **Runtime → Run all** and wait for completion.  
Estimated time with GPU: ~15-20 minutes.

## Results Summary

| Metric | Baseline CNN | Quality-Aware CNN |
|---|---|---|
| Accuracy | 96.2% | 96.2% |
| Precision | 0.9634 | 0.9872 |
| Recall | 0.9875 | 0.9625 |
| Specificity | 0.8800 | 0.9600 |
| F1-Score | 0.9753 | 0.9747 |
| ROC-AUC | 0.9950 | 0.9900 |
| False Negative | 1 | 3 |
| False Positive | 3 | 1 |

## Model Architecture
Custom CNN with 4 convolutional blocks:
- Block 1: Conv2d(3→32) + BatchNorm + ReLU + MaxPool
- Block 2: Conv2d(32→64) + BatchNorm + ReLU + MaxPool
- Block 3: Conv2d(64→128) + BatchNorm + ReLU + MaxPool
- Block 4: Conv2d(128→256) + BatchNorm + ReLU + MaxPool
- Classifier: Flatten → Linear(512) → Dropout(0.5) → Linear(1)
- Total parameters: 26,080,513

## Quality-Aware Strategy
Images are weighted during training using WeightedRandomSampler 
based on normalized quality scores, so higher-quality images 
are sampled more frequently during each epoch.

## References
- Abramovich et al. (2025). HYGD Dataset. PhysioNet. https://doi.org/10.13026/z0akkm33
- Kingma & Ba (2015). Adam Optimizer. https://arxiv.org/abs/1412.6980
- Ioffe & Szegedy (2015). Batch Normalization. https://arxiv.org/abs/1502.03167
- Srivastava et al. (2014). Dropout. http://jmlr.org/papers/v15/srivastava14a.html
- Selvaraju et al. (2017). Grad-CAM. https://arxiv.org/abs/1610.02391
- Buda et al. (2018). Class Imbalance in CNNs. https://doi.org/10.1016/j.neunet.2018.07.011
