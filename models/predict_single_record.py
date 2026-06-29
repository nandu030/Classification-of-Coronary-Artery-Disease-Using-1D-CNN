# === predict_single_record.py ===
import os
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from tensorflow.keras.models import load_model
import json, sys, io, base64
import matplotlib.pyplot as plt

# ======== CONFIG ========
BASE_DIR = r"C:\Users\nandi\OneDrive\Pictures\Desktop\PTB-CAD"
RECORDS_DIR = os.path.join(BASE_DIR, "records500")
EXTERNAL_DIR = os.path.join(BASE_DIR, "external_ecg")
MODEL_PATH = os.path.join(BASE_DIR, "paper_cnn_results", "best_cnn_model.keras")
META_CSV = os.path.join(BASE_DIR, "demo_records.csv")

TARGET_FS = 250
BANDPASS = (0.5, 40.0)
WINDOW_SAMPLES = 250

# ======== HELPERS ========
def bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, sig, axis=0)

def seg_to_windows(sig, fs_in, fs_target=250, window_samples=250):
    if fs_in != fs_target:
        n_target = int(round(sig.shape[0] * fs_target / fs_in))
        sig = resample(sig, n_target, axis=0)
    n_win = sig.shape[0] // window_samples
    return sig[:n_win * window_samples].reshape(n_win, window_samples, sig.shape[1])

def parse_true_label_from_hea(hea_path):
    """Extracts 'Normal' or 'CAD' from .hea comments if available."""
    true_label = "Unknown"
    try:
        with open(hea_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("#"):
                text = line.lower()
                if any(word in text for word in ["infarction", "ischemia", "stemi", "nstemi", "cad"]):
                    true_label = "CAD"
                    break
                elif any(word in text for word in ["healthy", "normal", "control"]):
                    true_label = "Normal"
                    break
    except Exception:
        pass
    return true_label

# ======== MAIN FUNCTION ========
def predict_single_record(record_name):
    if record_name.lower().startswith("s"):
        search_dir = EXTERNAL_DIR
        meta_available = False
    else:
        search_dir = RECORDS_DIR
        meta_available = True

    found = None
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.startswith(record_name) and f.endswith('.hea'):
                found = os.path.join(root, os.path.splitext(f)[0])
                break
        if found:
            break
    if not found:
        print(json.dumps({"error": f"Record {record_name} not found"}))
        sys.exit(1)

    # === Load record ===
    rec = wfdb.rdrecord(found)
    sig, fs = rec.p_signal, rec.fs

    # === Preprocess ===
    sig_f = bandpass_filter(sig, fs, *BANDPASS)
    windows = seg_to_windows(sig_f, fs, TARGET_FS, WINDOW_SAMPLES)
    windows = np.array([(w - np.mean(w, axis=0)) / (np.std(w, axis=0) + 1e-8)
                        for w in windows], dtype=np.float32)
    windows = np.mean(windows, axis=2, keepdims=False)

    # === Predict ===
    model = load_model(MODEL_PATH)
    preds = model.predict(windows, verbose=0)
    mean_pred = np.mean(preds)
    final_class = int(mean_pred > 0.5)
    label_map = {0: "Normal", 1: "CAD"}
    predicted_label = label_map[final_class]

    # === True label ===
    if meta_available:
        meta = pd.read_csv(META_CSV)
        row = meta[meta['filename_hr'].str.contains(record_name, na=False)].head(1)
        true_label = row.iloc[0]['True_Class_demo'] if not row.empty else "Unknown"
    else:
        hea_path = found + ".hea"
        true_label = parse_true_label_from_hea(hea_path)

    # === Visualization ===
    plt.figure(figsize=(14, 6), dpi=150)
    # Limit number of samples displayed for clarity
    max_samples = 3000
    sig_plot = sig[:max_samples, 0] if sig.shape[0] > max_samples else sig[:, 0]
    sig_f_plot = sig_f[:max_samples, 0] if sig_f.shape[0] > max_samples else sig_f[:, 0]

    plt.subplot(1, 2, 1)
    plt.plot(sig_plot, color='tomato')
    plt.title(f"Original ECG Signal")

    plt.subplot(1, 2, 2)
    plt.plot(sig_f_plot, color='seagreen')
    plt.title(f"Filtered ECG Signal")

    plt.suptitle(f"Prediction: {predicted_label} | True Label: {true_label}", fontsize=11, color='navy')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

    # === JSON Output ===
    result = {
        "record_name": record_name,
        "predicted_label": predicted_label,
        "true_label": true_label,
        "plot_image": plot_base64
    }

    print(json.dumps(result))

# ======== ENTRY POINT ========
if __name__ == "__main__":
    record_name = sys.argv[1] if len(sys.argv) > 1 else None
    if not record_name:
        print(json.dumps({"error": "No record name provided"}))
        sys.exit(1)
    predict_single_record(record_name)
