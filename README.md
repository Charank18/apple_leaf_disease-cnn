Apple Leaf Disease CNN with Lesion-Aware Explainability Validation

Overview

This repository contains a Convolutional Neural Network (CNN) implementation using the pre-trained ResNet50 model to classify apple leaf diseases (e.g., Healthy leaf, Alternaria leaf spot, Brown spot, Gray spot). The project includes a lesion-aware explainability validation framework using Grad-CAM (Gradient-weighted Class Activation Mapping) and Intersection over Union (IoU) metrics to assess the model's focus on lesion regions.

Setup

Prerequisites

Python 3.10

Git (for cloning the repository)

Installation

Clone the repository:
git clone https://github.com/your-username/apple-leaf-disease-cnn.git
cd apple-leaf-disease-cnn

Create a virtual environment:

python -m venv venv
.\venv\Scripts\activate  # On Windows
# or source venv/bin/activate  # On Unix/Linux/MacOS



Install dependencies:

pip install -r requirements.txt

Dataset Preparation





Place your dataset in a directory structure under dataset_root (e.g., C:\Users\chara\OneDrive\Desktop\cnn\apple-tree-leaf-disease-dataset\versions\1\):

Subdirectories for each class (e.g., Healthy leaf, Alternaria leaf spot).

Images (e.g., IMG_20190726_190843.jpg) in each class folder.

Optional masks in dataset_root/masks/<class_name>/ (e.g., IMG_20190726_190843_mask.png as binary images)


Update dataset_root in main.py to match your dataset path.

Usage

Activate the virtual environment (if not already active):

.\venv\Scripts\activate

Run the script:

python main.py



Check the results/ folder for:
Visualization files (e.g., <image_name>_visualization.png).

iou_results.csv with IoU scores and predictions.

Outputs

Visualizations: Grad-CAM heatmaps overlaid on original images, saved as PNG files.
Results: iou_results.csv containing columns: image, class, iou, top_prediction, top_score.

Average IoU: Printed to the console (may be low if masks are missing for diseased classes).

Files

main.py: Python script implementing the CNN classification and explainability validation.
requirements.txt: List of dependencies (e.g., TensorFlow, OpenCV, Matplotlib).
results/: Directory with output files (generated after running the script).

README.md: This file.

Notes





The current implementation uses dummy zero masks for healthy leaves and requires manual annotation of masks for diseased classes to improve IoU accuracy.



TensorFlow logs (e.g., oneDNN warnings) are normal and can be ignored unless performance optimization is needed.

Acknowledgments

This project was developed as part of the Lesion-Aware Explainability Validation task, utilizing the apple-tree-leaf-disease-dataset.

