# Classification of Coronary Artery Disease Using Feature Engineering and 1D-CNN

## Overview

This project presents an automated system for **Coronary Artery Disease (CAD)** classification using **12-lead Electrocardiogram (ECG)** signals from the **PTB-XL dataset**. The proposed approach combines ECG signal preprocessing, feature engineering, and a One-Dimensional Convolutional Neural Network (1D-CNN) to distinguish between **Normal** and **CAD** cases.

---

## Features

* ECG signal preprocessing using Butterworth band-pass filtering.
* Z-score normalization for signal standardization.
* ECG segmentation into fixed-length windows.
* Sample Entropy-based filtering to remove noisy and low-quality segments.
* Automatic feature learning using a lightweight 1D-CNN.
* Binary CAD classification (Normal vs CAD).
* Performance comparison with:

  * Logistic Regression
  * Support Vector Machine (RBF)
  * Random Forest
* Prediction support for both PTB-XL records and external ECG files.

---

## Dataset

**Dataset:** PTB-XL

* Total ECG records: **10,786**
* Sampling frequency: **500 Hz**
* ECG leads: **12**
* Labels generated using SCP diagnostic codes.

Binary mapping:

* **Normal:** NORM
* **CAD:** Myocardial Infarction (MI) and Ischemic (ISC/STTC-related) diagnostic codes

After preprocessing:

* Total ECG segments: **87,256**
* Normal segments: **77,048**
* CAD segments: **10,208**

---

## Project Workflow

1. Load PTB-XL ECG recordings.
2. Generate binary CAD labels from SCP diagnostic codes.
3. Apply Butterworth band-pass filtering.
4. Perform Z-score normalization.
5. Segment ECG signals into 250-sample windows.
6. Filter noisy segments using Sample Entropy.
7. Train the proposed 1D-CNN model.
8. Predict CAD/Normal class for unseen ECG records.
9. Evaluate using Accuracy, Sensitivity, Specificity, ROC-AUC, and Confusion Matrix.

---

## Folder Structure

```
Classification-of-Coronary-Artery-Disease-Using-1D-CNN/
│
├── README.md
├── preprocessing/
│   └── preprocess_paper_version.py
│
├── models/
│   ├── train_cnn_paper_model.py
│   └── predict_single_record.py
│
├── saved_models/
│   └── best_cnn_model.keras      
│
├── results/
│   ├── cnn_accuracy_curve.png
│   ├── cnn_confusion_matrix.png
│   ├── roc_comparison.png
│   ├── ml_model_comparison.csv
│   └── sample_prediction.png      
└── docs/
    ├── methodology.png            
    └── project_paper.pdf         
```

---

## Model Architecture

The proposed 1D-CNN consists of:

* Conv1D (32 filters)
* Batch Normalization
* MaxPooling
* Dropout
* Conv1D (64 filters)
* Batch Normalization
* MaxPooling
* Dropout
* Conv1D (128 filters)
* Batch Normalization
* MaxPooling
* Flatten
* Dense Layer
* Sigmoid Output Layer

Optimizer: Adam

Loss Function: Binary Cross-Entropy

---

## Performance

| Model               | Accuracy   | ROC-AUC   |
| ------------------- | ---------- | --------- |
| Logistic Regression | 90.16%     | 0.816     |
| SVM (RBF)           | 91.46%     | 0.825     |
| Random Forest       | 92.09%     | 0.891     |
| **Proposed 1D-CNN** | **95.09%** | **0.958** |

---

## Requirements

* Python 3.10+
* TensorFlow
* NumPy
* Pandas
* Scikit-learn
* SciPy
* Matplotlib
* Seaborn
* WFDB

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 – Preprocess ECG Signals

```bash
python preprocess_paper_version.py
```

### Step 2 – Train the CNN Model

```bash
python train_cnn_paper_model.py
```

### Step 3 – Predict on an ECG Record

```bash
python predict_single_record.py
```

Example input:

```
00001_hr
```

or for an external ECG:

```
s0010_re
```

---

## Results

The proposed 1D-CNN achieved superior performance compared with classical machine learning models while maintaining a lightweight architecture suitable for ECG signal classification.

---

## Future Work

* Multi-class cardiac disease classification.
* Real-time ECG monitoring.
* Clinical validation on larger datasets.
* Deployment as a web-based diagnostic support system.

---

## Author

**Ambati Nandini**

Bachelor of Technology (B.Tech)

Project: Classification of Coronary Artery Disease Using Feature Engineering and 1D-CNN
