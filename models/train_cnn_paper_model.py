# train_cnn_paper_model.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============ CONFIG ============
BASE_DIR = r"C:\Users\nandi\OneDrive\Pictures\Desktop\PTB-CAD"
DATA_DIR = os.path.join(BASE_DIR, "paper_preprocessed")
OUT_DIR = os.path.join(BASE_DIR, "paper_cnn_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ============ LOAD DATA ============
print("📂 Loading preprocessed data...")
X = np.load(os.path.join(DATA_DIR, "X_segments.npy"))
y = np.load(os.path.join(DATA_DIR, "y_labels.npy"))

X = X[..., np.newaxis]  # (samples, 250, 1)
X = X.astype(np.float32)
y = y.astype(int)

print(f"✅ Data loaded → X: {X.shape}, y: {y.shape}")

# Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============ DEFINE 1D CNN ============
def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        Conv1D(64, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_cnn_model((250, 1))
model.summary()

# ============ TRAIN MODEL ============
checkpoint_path = os.path.join(OUT_DIR, "best_cnn_model.keras")
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# ============ EVALUATE ============
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "CAD"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Normal", "CAD"], yticklabels=["Normal", "CAD"])
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cnn_confusion_matrix.png"))

# Save training curves
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cnn_accuracy_curve.png"))
plt.close()

print(f"📁 Model + plots saved to: {OUT_DIR}")
