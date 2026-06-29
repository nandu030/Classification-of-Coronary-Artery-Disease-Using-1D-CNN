# cnn_only_roc.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

BASE_DIR = r"C:\Users\nandi\OneDrive\Pictures\Desktop\PTB-CAD"
DATA_DIR = os.path.join(BASE_DIR, "paper_preprocessed")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "paper_cnn_results", "best_cnn_model.keras")

X = np.load(os.path.join(DATA_DIR, "X_segments.npy"))
y = np.load(os.path.join(DATA_DIR, "y_labels.npy"))

X = X[..., np.newaxis].astype(np.float32)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = load_model(CNN_MODEL_PATH)
probs = model.predict(X_test, verbose=0).ravel()

fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"1D-CNN (AUC={roc_auc:.3f})", linewidth=2)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Proposed 1D-CNN Model")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
