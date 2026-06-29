# === preprocess_paper_version.py ===
import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm

# ---------------- CONFIG ----------------
BASE_DIR = r"C:\Users\nandi\OneDrive\Pictures\Desktop\PTB-CAD"
RECORDS_DIR = os.path.join(BASE_DIR, "records500")
OUT_DIR = os.path.join(BASE_DIR, "paper_preprocessed")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_FS = 250
WINDOW_SIZE = 250
BANDPASS = (0.5, 40.0)

# ---------------- HELPERS ----------------
def bandpass_filter(sig, fs, lowcut=0.5, highcut=40.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, sig, axis=0)

def sample_entropy(x, m=2, r=0.1):
    N = len(x)
    def _phi(m):
        x_m = np.array([x[i:i+m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x_m[:, None, :] - x_m[None, :, :]), axis=2) <= r, axis=0)
        return np.sum(C)
    return -np.log(_phi(m+1) / _phi(m) + 1e-10)

def preprocess_record(filepath):
    rec = wfdb.rdrecord(filepath)
    sig, fs = rec.p_signal, rec.fs
    sig = bandpass_filter(sig, fs, *BANDPASS)

    # resample if needed
    if fs != TARGET_FS:
        sig = resample(sig, int(sig.shape[0] * TARGET_FS / fs), axis=0)

    # mean across all leads (paper used averaged 12-lead)
    mean_sig = np.mean(sig, axis=1)
    # normalize
    mean_sig = (mean_sig - np.mean(mean_sig)) / (np.std(mean_sig) + 1e-8)

    # segment into 1-second windows
    n_seg = len(mean_sig) // WINDOW_SIZE
    segments = mean_sig[:n_seg * WINDOW_SIZE].reshape(n_seg, WINDOW_SIZE)
    return segments


#Feature engineering filters out noisy, flat, or corrupted ECG segments

def feature_filter(segments, std_thresh=(0.1, 2.0), entropy_thresh=(0.1, 2.0)):
    valid_segments = []
    for seg in segments:
        sdev = np.std(seg)
        sent = sample_entropy(seg)
        if std_thresh[0] <= sdev <= std_thresh[1] and entropy_thresh[0] <= sent <= entropy_thresh[1]:
            valid_segments.append(seg)
    return np.array(valid_segments)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    all_segments, labels = [], []
    meta_file = os.path.join(BASE_DIR, "demo_records_with_predictions.csv")
    import pandas as pd
    df = pd.read_csv(meta_file)

    print(f"Loaded metadata: {len(df)} records")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        record_path = os.path.join(BASE_DIR, row["filename_hr"].replace("/", os.sep))
        if not os.path.exists(record_path + ".dat"):
            continue
        segs = preprocess_record(record_path)
        valid_segs = feature_filter(segs)
        if len(valid_segs) > 0:
            all_segments.append(valid_segs)
            labels.extend([row["True_Label_demo"]] * len(valid_segs))

    X = np.vstack(all_segments)
    y = np.array(labels)

    np.save(os.path.join(OUT_DIR, "X_segments.npy"), X)
    np.save(os.path.join(OUT_DIR, "y_labels.npy"), y)
    print(f"✅ Saved preprocessed segments → {OUT_DIR}")
    print("Shape:", X.shape, "Labels:", y.shape)
