# 🐟 Fish-classification

This repository provides a complete pipeline for building, training, and evaluating an image classification model for fish recognition using transfer learning.

## 📁 Project Structure

```
Fish-classification/
│
├── 1-Build your own image classification dataset/
│   └── Scripts and instructions to create a labeled dataset.
│
├── 2-Transfer learning to train your own image classification model/
│   └── Fine-tune pretrained models (e.g., ResNet, MobileNet) on your dataset.
│
├── 3-Evaluate the accuracy of the image classification algorithm/
    └── Tools to assess model performance using accuracy, precision, recall, etc.
```

## 🚀 Quick Start

### 1. Build Your Dataset

Prepare your fish image dataset with separate folders for each class:

```
dataset/
├── tilapia/
├── carp/
├── catfish/
└── ...
```

### 2. Train the Model

Run transfer learning scripts in folder `2-Transfer learning...` to fine-tune a pretrained model.

### 3. Evaluate Performance

Use the evaluation tools in `3-Evaluate...` to analyze classification results and visualize metrics.

## 🛠️ Requirements

* Python ≥ 3.7
* PyTorch ≥ 1.7
* torchvision
* OpenCV
* scikit-learn
  (Install all dependencies using `pip install -r requirements.txt`)



