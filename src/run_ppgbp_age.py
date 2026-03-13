"""
WearAge - First pipeline: PPG-BP age prediction using Pulse-PPG embeddings.

Downloads PPG-BP dataset (205 subjects), extracts Pulse-PPG embeddings,
and trains a Ridge regression to predict chronological age.
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
from scipy.signal import resample

# Add pulseppg to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "pulseppg"))

from pulseppg.nets.ResNet1D.ResNet1D_Net import Net

# ─── Config ───────────────────────────────────────────────────────────
PPGBP_URL = "https://figshare.com/ndownloader/articles/5459299/versions/5"
PPGBP_DIR = PROJECT_ROOT / "data" / "ppgbp"
MODEL_PATH = PROJECT_ROOT / "pulseppg" / "pulseppg" / "experiments" / "out" / "pulseppg" / "checkpoint_best.pkl"
FS_ORIGINAL = 1000  # PPG-BP original sampling rate
FS_TARGET = 50      # Pulse-PPG expected sampling rate
SEGMENT_LEN = 12000 # 4 minutes at 50Hz (Pulse-PPG default)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def download_ppgbp():
    """Download and extract PPG-BP dataset if not present."""
    if (PPGBP_DIR / "Data File").exists():
        print(f"PPG-BP already exists at {PPGBP_DIR}")
        return

    import requests
    import zipfile
    import io

    PPGBP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = PPGBP_DIR / "ppgbp.zip"

    print("Downloading PPG-BP dataset (~2.3 MB)...")
    r = requests.get(PPGBP_URL, stream=True)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(PPGBP_DIR)

    # Extract inner zip
    inner_zip = PPGBP_DIR / "PPG-BP Database.zip"
    if inner_zip.exists():
        with zipfile.ZipFile(inner_zip, "r") as z:
            z.extractall(PPGBP_DIR)
        inner_zip.unlink()
    zip_path.unlink()
    print("PPG-BP downloaded and extracted.")


def load_ppgbp_data():
    """Load PPG-BP subjects: returns (ppg_signals, ages, subject_ids)."""
    import pandas as pd
    from scipy.signal import butter, filtfilt

    excel_path = PPGBP_DIR / "Data File" / "PPG-BP dataset.xlsx"
    subject_dir = PPGBP_DIR / "Data File" / "0_subject"

    df = pd.read_excel(excel_path, header=1)
    df = df.rename(columns={
        "Age(year)": "age",
        "Sex(M/F)": "sex",
        "Systolic Blood Pressure(mmHg)": "sbp",
        "Diastolic Blood Pressure(mmHg)": "dbp",
        "Heart Rate(b/m)": "hr",
        "BMI(kg/m^2)": "bmi",
    })

    # Bandpass filter 0.5-8 Hz
    b, a = butter(4, [0.5, 8.0], btype='band', fs=FS_ORIGINAL)

    ppg_segments = []
    ages = []
    subject_ids = []
    extra_info = []  # sbp, dbp, bmi, sex

    filenames = sorted(set(f.split("_")[0] for f in os.listdir(subject_dir) if f.endswith(".txt")))
    print(f"Found {len(filenames)} subjects in PPG-BP")

    for fname in filenames:
        sid = int(fname)
        row = df[df["subject_ID"] == sid]
        if row.empty or pd.isna(row["age"].values[0]):
            continue

        age = float(row["age"].values[0])
        sbp = float(row["sbp"].values[0]) if not pd.isna(row["sbp"].values[0]) else np.nan
        dbp = float(row["dbp"].values[0]) if not pd.isna(row["dbp"].values[0]) else np.nan
        bmi = float(row["bmi"].values[0]) if not pd.isna(row["bmi"].values[0]) else np.nan
        sex = str(row["sex"].values[0])

        # Each subject has 3 segments
        segments_for_subject = []
        for seg_idx in range(1, 4):
            txt_path = subject_dir / f"{fname}_{seg_idx}.txt"
            if not txt_path.exists():
                continue
            with open(txt_path, 'r') as f:
                text = f.read().strip()
            values = [float(v) for v in text.split('\t') if v.strip()]
            signal = np.array(values, dtype=np.float64)

            # Filter
            signal = filtfilt(b, a, signal)

            # Resample to 50Hz
            n_samples_target = int(len(signal) * FS_TARGET / FS_ORIGINAL)
            signal = resample(signal, n_samples_target)

            # Z-score normalize
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

            segments_for_subject.append(signal)

        if not segments_for_subject:
            continue

        # Concatenate segments for this subject, then cut to SEGMENT_LEN
        full_signal = np.concatenate(segments_for_subject)
        if len(full_signal) < SEGMENT_LEN:
            # Pad with zeros if too short
            full_signal = np.pad(full_signal, (0, SEGMENT_LEN - len(full_signal)))
        else:
            full_signal = full_signal[:SEGMENT_LEN]

        ppg_segments.append(full_signal)
        ages.append(age)
        subject_ids.append(sid)
        extra_info.append({"sbp": sbp, "dbp": dbp, "bmi": bmi, "sex": sex})

    ppg_array = np.array(ppg_segments, dtype=np.float32)
    ages_array = np.array(ages, dtype=np.float32)
    print(f"Loaded {len(ages_array)} subjects, age range: {ages_array.min():.0f}-{ages_array.max():.0f}")
    return ppg_array, ages_array, subject_ids, extra_info


def load_pulseppg_model():
    """Load pre-trained Pulse-PPG model."""
    model = Net(
        in_channels=1,
        base_filters=128,
        kernel_size=11,
        stride=2,
        groups=1,
        n_block=12,
        finalpool="max"
    )

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["net"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    print(f"Pulse-PPG model loaded (epoch {ckpt['epoch']}), device: {DEVICE}")
    return model


def extract_embeddings(model, ppg_array):
    """Extract Pulse-PPG embeddings for all subjects."""
    import warnings
    warnings.filterwarnings("ignore", message="input's size at dim=1")

    embeddings = []
    batch_size = 16

    # Shape: (N, 1, SEGMENT_LEN) for the model
    X = torch.tensor(ppg_array, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size].to(DEVICE)
            emb = model(batch).cpu().numpy()
            embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Extracted embeddings: shape {embeddings.shape}")
    return embeddings


def train_and_evaluate(embeddings, ages, subject_ids):
    """Train Ridge regression on embeddings -> age, with leave-one-out or k-fold CV."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Standardize embeddings
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)
    y = ages

    # 5-fold cross-validation (subject-level, since each subject = 1 sample)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_mae = float("inf")
    best_alpha = 1.0

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha)
        y_pred = cross_val_predict(model, X, y, cv=kf)
        mae = mean_absolute_error(y, y_pred)
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha

    # Final evaluation with best alpha
    model = Ridge(alpha=best_alpha)
    y_pred = cross_val_predict(model, X, y, cv=kf)

    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    sdae = np.std(np.abs(y - y_pred))
    corr = np.corrcoef(y, y_pred)[0, 1]

    print("\n" + "="*60)
    print("RESULTS: Pulse-PPG embeddings -> Age prediction (PPG-BP)")
    print("="*60)
    print(f"  Best alpha:  {best_alpha}")
    print(f"  MAE:         {mae:.2f} years")
    print(f"  SDAE:        {sdae:.2f} years")
    print(f"  R2:          {r2:.4f}")
    print(f"  Correlation: {corr:.4f}")
    print(f"  N subjects:  {len(y)}")
    print(f"  Age range:   {y.min():.0f}-{y.max():.0f}")
    print("="*60)

    # Comparison baseline: predict mean age for everyone
    baseline_mae = mean_absolute_error(y, np.full_like(y, y.mean()))
    print(f"\n  Baseline (predict mean): MAE = {baseline_mae:.2f} years")
    print(f"  Improvement over baseline: {baseline_mae - mae:.2f} years ({(baseline_mae - mae)/baseline_mae*100:.1f}%)")

    return y_pred, best_alpha


def save_results(ages, y_pred, subject_ids, extra_info):
    """Save predictions and make a simple plot."""
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Save CSV
    import csv
    with open(results_dir / "ppgbp_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "true_age", "predicted_age", "age_gap", "sbp", "dbp", "bmi", "sex"])
        for i, sid in enumerate(subject_ids):
            gap = y_pred[i] - ages[i]
            info = extra_info[i]
            writer.writerow([sid, f"{ages[i]:.1f}", f"{y_pred[i]:.1f}", f"{gap:.1f}",
                           f"{info['sbp']:.0f}" if not np.isnan(info['sbp']) else "",
                           f"{info['dbp']:.0f}" if not np.isnan(info['dbp']) else "",
                           f"{info['bmi']:.1f}" if not np.isnan(info['bmi']) else "",
                           info['sex']])
    print(f"\nPredictions saved to {results_dir / 'ppgbp_predictions.csv'}")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter: true vs predicted
        ax = axes[0]
        ax.scatter(ages, y_pred, alpha=0.6, s=30)
        ax.plot([ages.min(), ages.max()], [ages.min(), ages.max()], 'r--', label='Perfect prediction')
        ax.set_xlabel('Chronological Age (years)')
        ax.set_ylabel('Predicted Age (years)')
        ax.set_title('Pulse-PPG Age Prediction on PPG-BP')
        ax.legend()

        # Bland-Altman
        ax = axes[1]
        mean_ages = (ages + y_pred) / 2
        diff = y_pred - ages
        ax.scatter(mean_ages, diff, alpha=0.6, s=30)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.axhline(y=np.mean(diff) + 1.96*np.std(diff), color='gray', linestyle=':', label='+1.96 SD')
        ax.axhline(y=np.mean(diff) - 1.96*np.std(diff), color='gray', linestyle=':', label='-1.96 SD')
        ax.set_xlabel('Mean of True and Predicted Age')
        ax.set_ylabel('Predicted - True Age (years)')
        ax.set_title('Bland-Altman Plot')
        ax.legend()

        plt.tight_layout()
        plt.savefig(results_dir / "ppgbp_age_prediction.png", dpi=150)
        print(f"Plot saved to {results_dir / 'ppgbp_age_prediction.png'}")
    except Exception as e:
        print(f"Could not create plot: {e}")


def main():
    print("="*60)
    print("WearAge - PPG-BP Age Prediction Pipeline")
    print("="*60)

    # Step 1: Download PPG-BP
    print("\n[1/4] Downloading PPG-BP dataset...")
    download_ppgbp()

    # Step 2: Load data
    print("\n[2/4] Loading PPG-BP data with age labels...")
    ppg_array, ages, subject_ids, extra_info = load_ppgbp_data()

    # Step 3: Extract embeddings
    print("\n[3/4] Extracting Pulse-PPG embeddings...")
    model = load_pulseppg_model()
    embeddings = extract_embeddings(model, ppg_array)

    # Step 4: Train and evaluate
    print("\n[4/4] Training Ridge regression (age prediction)...")
    y_pred, best_alpha = train_and_evaluate(embeddings, ages, subject_ids)

    # Save results
    save_results(ages, y_pred, subject_ids, extra_info)

    print("\nDone.")


if __name__ == "__main__":
    main()
