"""
PulseDB Age Prediction Pipeline
Load VitalDB segments -> extract embeddings (PaPaGei + Pulse-PPG) -> Ridge regression -> evaluate.
"""

import sys
import os
import numpy as np
import h5py
from pathlib import Path
from scipy.signal import resample, find_peaks

PROJECT_ROOT = Path(__file__).parent.parent

# ── Config ──────────────────────────────────────────────────────────────────
FS_PULSEDB = 125        # PulseDB native sampling rate
FS_PULSEPPG = 50        # Pulse-PPG expected input rate
SEGMENT_LEN_125 = 1250  # 10s at 125Hz (PaPaGei native)
SEGMENT_LEN_50 = 500    # 10s at 50Hz (Pulse-PPG)
MIN_AGE = 18            # Filter out pediatric subjects
MAX_SEGMENTS_PER_SUBJECT = 50  # Cap segments per subject to avoid imbalance


def extract_hr_hrv_features(signals_125, fs=125):
    """Extract HR and HRV features from PPG signals via peak detection.

    For each 10s PPG segment, detect systolic peaks, compute inter-beat intervals (IBI),
    then derive HR and HRV metrics.

    Returns:
        features: np.array (N, 7) - [mean_HR, std_HR, SDNN, RMSSD, pNN50, mean_IBI, range_HR]
        valid_mask: np.array (N,) bool - True if enough peaks were found
    """
    n = len(signals_125)
    features = np.full((n, 7), np.nan, dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        sig = signals_125[i]

        # Detect systolic peaks: min distance ~0.4s (150 BPM max), prominence > 0.1
        peaks, properties = find_peaks(sig, distance=int(fs * 0.4),
                                        prominence=0.1, height=0.3)

        if len(peaks) < 4:
            # Not enough peaks for meaningful HRV
            continue

        # Inter-beat intervals in seconds
        ibi = np.diff(peaks) / fs

        # Filter physiologically plausible IBIs (30-200 BPM -> 0.3-2.0s)
        ibi_valid = ibi[(ibi >= 0.3) & (ibi <= 2.0)]
        if len(ibi_valid) < 3:
            continue

        # HR features (bpm)
        hr = 60.0 / ibi_valid
        mean_hr = np.mean(hr)
        std_hr = np.std(hr)
        range_hr = np.max(hr) - np.min(hr)

        # HRV features (in ms for clinical convention, but we store in seconds for consistency)
        sdnn = np.std(ibi_valid)             # Standard deviation of NN intervals
        successive_diff = np.diff(ibi_valid)
        rmssd = np.sqrt(np.mean(successive_diff ** 2))  # Root mean square of successive differences
        pnn50 = np.sum(np.abs(successive_diff) > 0.05) / len(successive_diff)  # % successive diffs > 50ms
        mean_ibi = np.mean(ibi_valid)

        features[i] = [mean_hr, std_hr, sdnn, rmssd, pnn50, mean_ibi, range_hr]
        valid_mask[i] = True

    pct = 100 * valid_mask.sum() / n
    print(f"  HR/HRV extraction: {valid_mask.sum()}/{n} segments valid ({pct:.1f}%)")
    return features, valid_mask


HR_HRV_FEATURE_NAMES = ['mean_HR', 'std_HR', 'SDNN', 'RMSSD', 'pNN50', 'mean_IBI', 'range_HR']


def load_pulsedb_vitaldb(data_dir, max_files=None):
    """Load PPG segments from VitalDB .mat files.

    Returns:
        signals_125: np.array (N, 1250) - PPG at 125Hz
        ages: np.array (N,) - chronological age
        subject_ids: list of str - subject ID per segment
        demographics: dict with 'gender', 'bmi', 'height', 'weight', 'sbp', 'dbp' per segment
    """
    mat_dir = Path(data_dir)
    mat_files = sorted(mat_dir.glob("p*.mat"))
    if max_files:
        mat_files = mat_files[:max_files]

    all_signals = []
    all_ages = []
    all_subjects = []
    all_demo = {'gender': [], 'bmi': [], 'height': [], 'weight': [], 'sbp': [], 'dbp': []}

    def _read_float(f, sw, key, idx):
        try:
            ref = sw[key][0, idx]
            return float(f[ref][0, 0])
        except Exception:
            return np.nan

    for fpath in mat_files:
        try:
            f = h5py.File(str(fpath), 'r')
            sw = f['Subj_Wins']
            n_segs = sw['Age'].shape[1]

            # Get age from first segment (same for all segments of a subject)
            ref = sw['Age'][0, 0]
            age = f[ref][0, 0]

            if age < MIN_AGE or np.isnan(age):
                f.close()
                continue

            # Read subject-level demographics from first segment
            gender_code = _read_float(f, sw, 'Gender', 0)
            bmi = _read_float(f, sw, 'BMI', 0)
            height = _read_float(f, sw, 'Height', 0)
            weight = _read_float(f, sw, 'Weight', 0)

            # Limit segments per subject
            n_use = min(n_segs, MAX_SEGMENTS_PER_SUBJECT)
            # Sample evenly if we need to subsample
            indices = np.linspace(0, n_segs - 1, n_use, dtype=int)

            subject_id = fpath.stem
            for i in indices:
                ref = sw['PPG_F'][0, i]
                ppg = f[ref][:].flatten().astype(np.float32)
                if len(ppg) == SEGMENT_LEN_125:
                    all_signals.append(ppg)
                    all_ages.append(age)
                    all_subjects.append(subject_id)
                    # Gender: M=77->1, F=70->0
                    all_demo['gender'].append(1.0 if gender_code == 77 else 0.0)
                    all_demo['bmi'].append(bmi)
                    all_demo['height'].append(height)
                    all_demo['weight'].append(weight)
                    # SBP/DBP per segment (varies across segments)
                    all_demo['sbp'].append(_read_float(f, sw, 'SegSBP', i))
                    all_demo['dbp'].append(_read_float(f, sw, 'SegDBP', i))

            f.close()
            print(f"  {subject_id}: age={age:.0f}, loaded {n_use}/{n_segs} segments")
        except Exception as e:
            print(f"  Error loading {fpath.name}: {e}")

    signals = np.array(all_signals, dtype=np.float32)
    ages = np.array(all_ages, dtype=np.float32)
    demographics = {k: np.array(v, dtype=np.float32) for k, v in all_demo.items()}
    print(f"\nTotal: {len(ages)} segments from {len(set(all_subjects))} subjects")
    return signals, ages, all_subjects, demographics


def load_papagei_model():
    """Load PaPaGei-S model."""
    import torch
    sys.path.insert(0, str(PROJECT_ROOT / "papagei-foundation-model"))
    from models.resnet import ResNet1DMoE

    model_config = {
        'base_filters': 32, 'kernel_size': 3, 'stride': 2,
        'groups': 1, 'n_block': 18, 'n_classes': 512, 'n_experts': 3
    }

    model = ResNet1DMoE(
        in_channels=1,
        base_filters=model_config['base_filters'],
        kernel_size=model_config['kernel_size'],
        stride=model_config['stride'],
        groups=model_config['groups'],
        n_block=model_config['n_block'],
        n_classes=model_config['n_classes'],
        n_experts=model_config['n_experts']
    )

    weights_path = PROJECT_ROOT / "papagei-foundation-model" / "weights" / "papagei_s.pt"
    state_dict = torch.load(str(weights_path), map_location='cpu', weights_only=True)
    # Remove 'module.' prefix if present (from DataParallel)
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace('module.', '')] = v
    model.load_state_dict(cleaned, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"PaPaGei-S loaded on {device}")
    return model, device


def load_pulseppg_model():
    """Load Pulse-PPG model."""
    import torch
    sys.path.insert(0, str(PROJECT_ROOT / "pulseppg"))
    from pulseppg.nets.ResNet1D.ResNet1D_Net import Net as ResNet1D

    model = ResNet1D(
        in_channels=1, base_filters=128, kernel_size=11,
        stride=2, groups=1, n_block=12,
        downsample_gap=2, increasefilter_gap=4, use_do=False,
        finalpool="max"
    )

    weights_path = PROJECT_ROOT / "pulseppg" / "pulseppg" / "experiments" / "out" / "pulseppg" / "checkpoint_best.pkl"
    checkpoint = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    state_dict = checkpoint['net']
    cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Pulse-PPG loaded on {device}")
    return model, device


def load_aippg_model():
    """Load AI-PPG Age (VascularAge) model from Ngks03/PPG-VascularAge.

    This model takes averaged single-beat PPG waveforms and directly predicts vascular age.
    Architecture: Net1D with SE blocks, trained on UK Biobank with Dist loss.
    """
    import torch
    import json
    sys.path.insert(0, str(PROJECT_ROOT / "ppg-vascularage"))
    from net1d import Net1D

    config_path = PROJECT_ROOT / "ppg-vascularage" / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    model = Net1D(**cfg)

    weights_path = PROJECT_ROOT / "ppg-vascularage" / "model.pth"
    state_dict = torch.load(str(weights_path), map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"AI-PPG Age loaded on {device} ({sum(p.numel() for p in model.parameters())/1e3:.0f}K params)")
    return model, device


BEAT_RESAMPLE_LEN = 100  # Resample each beat to 100 samples (UK Biobank convention)


def extract_beat_templates(signals_125, fs=125):
    """Extract averaged beat templates from 10s PPG segments.

    For each segment: detect systolic peaks, segment individual beats (peak-to-peak),
    resample each beat to BEAT_RESAMPLE_LEN samples, average across beats.

    Returns:
        templates: np.array (N, BEAT_RESAMPLE_LEN) - averaged beat templates
        valid_mask: np.array (N,) bool - True if enough beats were found
    """
    n = len(signals_125)
    templates = np.zeros((n, BEAT_RESAMPLE_LEN), dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        sig = signals_125[i]
        peaks, _ = find_peaks(sig, distance=int(fs * 0.4), prominence=0.1, height=0.3)

        if len(peaks) < 4:
            continue

        # Extract individual beats (peak-to-peak)
        beats = []
        for j in range(len(peaks) - 1):
            beat = sig[peaks[j]:peaks[j+1]]
            beat_len = len(beat)
            # Filter physiologically plausible beat lengths (0.3-2.0s)
            if beat_len < int(fs * 0.3) or beat_len > int(fs * 2.0):
                continue
            # Resample to fixed length
            beat_resampled = resample(beat, BEAT_RESAMPLE_LEN)
            beats.append(beat_resampled)

        if len(beats) < 3:
            continue

        # Average across beats to create template
        template = np.mean(beats, axis=0).astype(np.float32)
        templates[i] = template
        valid_mask[i] = True

    pct = 100 * valid_mask.sum() / n
    print(f"  Beat template extraction: {valid_mask.sum()}/{n} segments valid ({pct:.1f}%)")
    return templates, valid_mask


def predict_age_aippg(model, device, templates, batch_size=256):
    """Predict vascular age directly using AI-PPG Age model.

    The model outputs a single age value per beat template (no Ridge regression needed).
    """
    import torch

    predictions = []
    n = len(templates)
    for i in range(0, n, batch_size):
        batch = templates[i:i+batch_size]
        # z-score normalize per beat
        batch = (batch - batch.mean(axis=1, keepdims=True)) / (batch.std(axis=1, keepdims=True) + 1e-8)
        tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).to(device)  # (B, 1, 100)
        with torch.inference_mode():
            out = model(tensor).cpu().numpy().squeeze(-1)  # (B,)
        predictions.append(out)

    return np.concatenate(predictions, axis=0)


def extract_embeddings_aippg(model, device, templates, batch_size=256):
    """Extract penultimate-layer embeddings from AI-PPG Age model.

    Hooks into the model before the final dense layer to get feature representations.
    """
    import torch

    embeddings = []
    n = len(templates)

    # Register hook on the global average pooling (before dense layer)
    hook_output = {}
    def hook_fn(module, input, output):
        hook_output['emb'] = output

    # The last stage output goes through mean(-1), then dense
    # We hook the last stage to capture pre-dense features
    last_stage = model.stage_list[-1]
    handle = last_stage.register_forward_hook(hook_fn)

    for i in range(0, n, batch_size):
        batch = templates[i:i+batch_size]
        batch = (batch - batch.mean(axis=1, keepdims=True)) / (batch.std(axis=1, keepdims=True) + 1e-8)
        tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.inference_mode():
            model(tensor)
            # hook_output['emb'] is (B, C, L) from last stage; apply same mean pooling
            emb = hook_output['emb'].mean(-1).cpu().numpy()  # (B, C)
        embeddings.append(emb)

    handle.remove()
    return np.concatenate(embeddings, axis=0)


def extract_embeddings_papagei(model, device, signals_125, batch_size=32):
    """Extract 512-dim embeddings using PaPaGei-S (125Hz native)."""
    import torch

    embeddings = []
    n = len(signals_125)
    for i in range(0, n, batch_size):
        batch = signals_125[i:i+batch_size]
        # z-score normalize per segment
        batch = (batch - batch.mean(axis=1, keepdims=True)) / (batch.std(axis=1, keepdims=True) + 1e-8)
        tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).to(device)  # (B, 1, 1250)
        with torch.inference_mode():
            outputs = model(tensor)
            emb = outputs[0].cpu().numpy()  # (B, 512)
        embeddings.append(emb)

    return np.concatenate(embeddings, axis=0)


def extract_embeddings_pulseppg(model, device, signals_125, batch_size=32):
    """Extract 512-dim embeddings using Pulse-PPG (resample to 50Hz).

    Pulse-PPG Net with finalpool='max' returns 512-dim embeddings directly.
    """
    import torch

    # Resample from 125Hz to 50Hz
    signals_50 = np.array([
        resample(s, SEGMENT_LEN_50) for s in signals_125
    ], dtype=np.float32)

    embeddings = []
    n = len(signals_50)
    for i in range(0, n, batch_size):
        batch = signals_50[i:i+batch_size]
        batch = (batch - batch.mean(axis=1, keepdims=True)) / (batch.std(axis=1, keepdims=True) + 1e-8)
        tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).to(device)  # (B, 1, 500)
        with torch.inference_mode():
            emb = model(tensor).cpu().numpy()  # (B, 512)
        embeddings.append(emb)

    return np.concatenate(embeddings, axis=0)


def train_and_evaluate(embeddings, ages, subject_ids, model_name):
    """Subject-level split, Ridge regression, evaluate."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score

    # Subject-level train/test split (80/20)
    unique_subjects = list(set(subject_ids))
    np.random.seed(42)
    np.random.shuffle(unique_subjects)
    n_train = int(len(unique_subjects) * 0.8)
    train_subjects = set(unique_subjects[:n_train])
    test_subjects = set(unique_subjects[n_train:])

    train_mask = np.array([s in train_subjects for s in subject_ids])
    test_mask = ~train_mask

    X_train, y_train = embeddings[train_mask], ages[train_mask]
    X_test, y_test = embeddings[test_mask], ages[test_mask]

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")
    print(f"  Train: {len(X_train)} segments from {n_train} subjects")
    print(f"  Test:  {len(X_test)} segments from {len(test_subjects)} subjects")

    # Scale + Ridge
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
    ridge.fit(X_train_s, y_train)

    y_pred = ridge.predict(X_test_s)

    # Segment-level metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0

    print(f"\n  Segment-level results:")
    print(f"    MAE:  {mae:.2f} years")
    print(f"    R2:   {r2:.3f}")
    print(f"    Corr: {corr:.3f}")
    print(f"    Best alpha: {ridge.alpha_}")

    # Subject-level aggregation (mean prediction per subject)
    test_subject_list = [s for s, m in zip(subject_ids, test_mask) if m]
    subj_preds = {}
    subj_trues = {}
    for s, pred, true in zip(test_subject_list, y_pred, y_test):
        subj_preds.setdefault(s, []).append(pred)
        subj_trues.setdefault(s, []).append(true)

    subj_pred_mean = np.array([np.mean(subj_preds[s]) for s in subj_preds])
    subj_true_mean = np.array([subj_trues[s][0] for s in subj_trues])

    subj_mae = mean_absolute_error(subj_true_mean, subj_pred_mean)
    subj_r2 = r2_score(subj_true_mean, subj_pred_mean) if len(subj_true_mean) > 1 else 0
    subj_corr = np.corrcoef(subj_true_mean, subj_pred_mean)[0, 1] if len(subj_true_mean) > 1 else 0

    print(f"\n  Subject-level results (averaged predictions):")
    print(f"    MAE:  {subj_mae:.2f} years")
    print(f"    R2:   {subj_r2:.3f}")
    print(f"    Corr: {subj_corr:.3f}")

    # Baseline: predict mean age
    baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))
    print(f"\n  Baseline MAE (predict train mean): {baseline_mae:.2f} years")

    return {
        'model': model_name,
        'seg_mae': mae, 'seg_r2': r2, 'seg_corr': corr,
        'subj_mae': subj_mae, 'subj_r2': subj_r2, 'subj_corr': subj_corr,
        'baseline_mae': baseline_mae,
        'y_test': y_test, 'y_pred': y_pred,
        'subj_true': subj_true_mean, 'subj_pred': subj_pred_mean,
    }


def train_and_evaluate_kfold(embeddings, ages, subject_ids, model_name, n_folds=5):
    """Stratified K-fold cross-validation at the subject level."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score

    # Build subject-level data: map subject -> (indices, age)
    subj_map = {}
    for i, (s, a) in enumerate(zip(subject_ids, ages)):
        subj_map.setdefault(s, {'indices': [], 'age': a})['indices'].append(i)

    unique_subjects = list(subj_map.keys())
    subj_ages = np.array([subj_map[s]['age'] for s in unique_subjects])

    # Stratify by age decade
    decade_bins = (subj_ages // 10).astype(int)
    from collections import defaultdict
    decade_groups = defaultdict(list)
    for i, d in enumerate(decade_bins):
        decade_groups[d].append(i)

    # Assign folds in round-robin within each decade
    np.random.seed(42)
    fold_assignments = np.zeros(len(unique_subjects), dtype=int)
    for decade, indices in decade_groups.items():
        np.random.shuffle(indices)
        for j, idx in enumerate(indices):
            fold_assignments[idx] = j % n_folds

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name} ({n_folds}-fold stratified CV)")
    print(f"{'='*60}")
    print(f"  Total: {len(ages)} segments from {len(unique_subjects)} subjects")

    fold_maes = []
    fold_subj_maes = []
    fold_corrs = []
    all_subj_true = []
    all_subj_pred = []

    for fold in range(n_folds):
        test_subj = set(s for s, f in zip(unique_subjects, fold_assignments) if f == fold)
        train_subj = set(unique_subjects) - test_subj

        train_mask = np.array([s in train_subj for s in subject_ids])
        test_mask = ~train_mask

        X_train, y_train = embeddings[train_mask], ages[train_mask]
        X_test, y_test = embeddings[test_mask], ages[test_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
        ridge.fit(X_train_s, y_train)
        y_pred = ridge.predict(X_test_s)

        seg_mae = mean_absolute_error(y_test, y_pred)
        fold_maes.append(seg_mae)

        # Subject-level aggregation
        test_subj_list = [s for s, m in zip(subject_ids, test_mask) if m]
        sp, st = {}, {}
        for s, p, t in zip(test_subj_list, y_pred, y_test):
            sp.setdefault(s, []).append(p)
            st.setdefault(s, []).append(t)

        subj_pred = np.array([np.mean(sp[s]) for s in sp])
        subj_true = np.array([st[s][0] for s in st])
        subj_mae = mean_absolute_error(subj_true, subj_pred)
        subj_corr = np.corrcoef(subj_true, subj_pred)[0, 1] if len(subj_true) > 2 else 0

        fold_subj_maes.append(subj_mae)
        fold_corrs.append(subj_corr)
        all_subj_true.extend(subj_true)
        all_subj_pred.extend(subj_pred)

        print(f"  Fold {fold+1}: {len(test_subj)} subj, seg MAE={seg_mae:.2f}, subj MAE={subj_mae:.2f}, corr={subj_corr:.3f}")

    # Overall metrics
    all_subj_true = np.array(all_subj_true)
    all_subj_pred = np.array(all_subj_pred)
    overall_mae = mean_absolute_error(all_subj_true, all_subj_pred)
    overall_r2 = r2_score(all_subj_true, all_subj_pred)
    overall_corr = np.corrcoef(all_subj_true, all_subj_pred)[0, 1]
    baseline_mae = np.mean(np.abs(all_subj_true - np.mean(all_subj_true)))

    print(f"\n  Cross-validated results (subject-level):")
    print(f"    MAE:  {overall_mae:.2f} +/- {np.std(fold_subj_maes):.2f} years")
    print(f"    R2:   {overall_r2:.3f}")
    print(f"    Corr: {overall_corr:.3f} +/- {np.std(fold_corrs):.3f}")
    print(f"    Baseline MAE (predict mean): {baseline_mae:.2f} years")

    # MAE by age decade
    print(f"\n  MAE by age decade:")
    decades = [(20, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]
    for low, high in decades:
        mask = (all_subj_true >= low) & (all_subj_true < high)
        if mask.sum() > 0:
            dec_mae = mean_absolute_error(all_subj_true[mask], all_subj_pred[mask])
            print(f"    {low}-{high}: MAE={dec_mae:.2f} (n={mask.sum()})")

    return {
        'model': model_name,
        'subj_mae': overall_mae, 'subj_mae_std': np.std(fold_subj_maes),
        'subj_r2': overall_r2, 'subj_corr': overall_corr,
        'baseline_mae': baseline_mae,
        'fold_maes': fold_subj_maes, 'fold_corrs': fold_corrs,
        'subj_true': all_subj_true, 'subj_pred': all_subj_pred,
    }


def save_results(results_list, output_dir):
    """Save comparison table and plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Model':<20} {'Subj MAE':>12} {'Subj R2':>8} {'Subj Corr':>11} {'Baseline':>9}")
    print(f"{'='*80}")
    for r in results_list:
        mae_str = f"{r['subj_mae']:.2f}"
        if 'subj_mae_std' in r:
            mae_str += f" +/- {r['subj_mae_std']:.2f}"
        print(f"{r['model']:<20} {mae_str:>12} {r['subj_r2']:>8.3f} {r['subj_corr']:>11.3f} {r['baseline_mae']:>9.2f}")

    # Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_models = len(results_list)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        for ax, r in zip(axes, results_list):
            ax.scatter(r['subj_true'], r['subj_pred'], alpha=0.7, s=40)
            lims = [min(r['subj_true'].min(), r['subj_pred'].min()) - 5,
                    max(r['subj_true'].max(), r['subj_pred'].max()) + 5]
            ax.plot(lims, lims, 'r--', lw=1)
            ax.set_xlabel('Chronological Age')
            ax.set_ylabel('Predicted Age')
            ax.set_title(f"{r['model']}\nMAE={r['subj_mae']:.1f}yr, R2={r['subj_r2']:.2f}")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect('equal')

        plt.tight_layout()
        plot_path = output_dir / "pulsedb_age_comparison.png"
        plt.savefig(str(plot_path), dpi=150)
        print(f"\nPlot saved: {plot_path}")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plots")


def evaluate_demographic_baseline(demographics, ages, subjects, output_dir):
    """Evaluate age prediction from demographics alone (sex, BMI, height, weight, BP).

    This provides context: how much does PPG add beyond simple demographics?
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score

    results = []

    # Build feature sets
    feature_sets = {
        'Sex only': np.array(demographics['gender']).reshape(-1, 1),
        'Sex + BMI': np.column_stack([demographics['gender'], demographics['bmi']]),
        'Sex + BMI + Height + Weight': np.column_stack([
            demographics['gender'], demographics['bmi'],
            demographics['height'], demographics['weight']
        ]),
        'All demographics': np.column_stack([
            demographics['gender'], demographics['bmi'],
            demographics['height'], demographics['weight'],
            demographics['sbp'], demographics['dbp']
        ]),
    }

    print(f"\n{'='*60}")
    print(f"DEMOGRAPHIC BASELINES (5-fold stratified CV)")
    print(f"{'='*60}")

    for feat_name, X in feature_sets.items():
        # Drop rows with NaN
        valid = ~np.any(np.isnan(X), axis=1)
        X_valid = X[valid]
        y_valid = ages[valid]
        subj_valid = [s for s, v in zip(subjects, valid) if v]

        if len(X_valid) < 50:
            print(f"  {feat_name}: not enough valid samples ({len(X_valid)}), skipping")
            continue

        r = train_and_evaluate_kfold(X_valid, y_valid, subj_valid, f"Demo: {feat_name}")
        results.append(r)

    return results


def main():
    data_dir = PROJECT_ROOT / "data" / "pulsedb_segments" / "PulseDB_Vital"
    output_dir = PROJECT_ROOT / "results" / "pulsedb_vitaldb"

    if not data_dir.exists() or not list(data_dir.glob("p*.mat")):
        print(f"No VitalDB segment files found in {data_dir}")
        return

    n_files = len(list(data_dir.glob("p*.mat")))
    print(f"Found {n_files} VitalDB segment files")

    # Load data
    print("\n--- Loading PulseDB VitalDB segments ---")
    signals, ages, subjects, demographics = load_pulsedb_vitaldb(data_dir)

    if len(signals) == 0:
        print("No valid segments loaded!")
        return

    results = []
    emb_papagei = None
    emb_pulseppg = None

    # Demographic baselines
    print("\n--- Demographic baselines ---")
    demo_results = evaluate_demographic_baseline(demographics, ages, subjects, output_dir)
    results.extend(demo_results)

    # HR/HRV baselines
    print("\n--- Extracting HR/HRV features ---")
    hr_features, hr_valid = extract_hr_hrv_features(signals)

    # HR only baseline
    hr_only = hr_features[hr_valid][:, [0]]  # mean_HR only
    r = train_and_evaluate_kfold(hr_only, ages[hr_valid],
                                  [s for s, v in zip(subjects, hr_valid) if v], "HR only")
    results.append(r)

    # HR + HRV baseline (all 7 features)
    hr_all = hr_features[hr_valid]
    r = train_and_evaluate_kfold(hr_all, ages[hr_valid],
                                  [s for s, v in zip(subjects, hr_valid) if v], "HR + HRV")
    results.append(r)

    # HR/HRV + Demographics
    demo_feats_hr = np.column_stack([
        demographics['gender'][hr_valid], demographics['bmi'][hr_valid],
        demographics['height'][hr_valid], demographics['weight'][hr_valid],
        demographics['sbp'][hr_valid], demographics['dbp'][hr_valid]
    ])
    valid_both = ~np.any(np.isnan(demo_feats_hr), axis=1)
    if valid_both.sum() > 50:
        hr_demo_fused = np.column_stack([hr_all[valid_both], demo_feats_hr[valid_both]])
        subj_both = [s for s, v in zip(
            [s for s, v in zip(subjects, hr_valid) if v], valid_both) if v]
        ages_both = ages[hr_valid][valid_both]
        r = train_and_evaluate_kfold(hr_demo_fused, ages_both, subj_both, "HR/HRV + Demo")
        results.append(r)

    # PaPaGei (125Hz native - no resampling needed)
    print("\n--- Extracting PaPaGei embeddings ---")
    try:
        papagei_model, device = load_papagei_model()
        emb_papagei = extract_embeddings_papagei(papagei_model, device, signals)
        print(f"PaPaGei embeddings: {emb_papagei.shape}")
        del papagei_model  # Free memory
        r = train_and_evaluate_kfold(emb_papagei, ages, subjects, "PaPaGei-S")
        results.append(r)
    except Exception as e:
        print(f"PaPaGei failed: {e}")
        import traceback; traceback.print_exc()

    # Pulse-PPG (resample to 50Hz)
    print("\n--- Extracting Pulse-PPG embeddings ---")
    try:
        pulseppg_model, device = load_pulseppg_model()
        emb_pulseppg = extract_embeddings_pulseppg(pulseppg_model, device, signals)
        print(f"Pulse-PPG embeddings: {emb_pulseppg.shape}")
        del pulseppg_model
        r = train_and_evaluate_kfold(emb_pulseppg, ages, subjects, "Pulse-PPG")
        results.append(r)
    except Exception as e:
        print(f"Pulse-PPG failed: {e}")
        import traceback; traceback.print_exc()

    # AI-PPG Age (direct age predictor + embeddings)
    emb_aippg = None
    print("\n--- AI-PPG Age (VascularAge) ---")
    try:
        aippg_model, device = load_aippg_model()
        print("  Extracting beat templates...")
        beat_templates, beat_valid = extract_beat_templates(signals)

        if beat_valid.sum() > 100:
            # 1) Direct zero-shot prediction (no training needed)
            print("  Running direct age prediction (zero-shot)...")
            direct_preds = predict_age_aippg(aippg_model, device, beat_templates[beat_valid])
            # Evaluate direct predictions with subject-level aggregation
            from sklearn.metrics import mean_absolute_error, r2_score
            beat_ages = ages[beat_valid]
            beat_subjects = [s for s, v in zip(subjects, beat_valid) if v]
            sp, st = {}, {}
            for s, p, t in zip(beat_subjects, direct_preds, beat_ages):
                sp.setdefault(s, []).append(p)
                st.setdefault(s, []).append(t)
            subj_pred = np.array([np.mean(sp[s]) for s in sp])
            subj_true = np.array([st[s][0] for s in st])
            direct_mae = mean_absolute_error(subj_true, subj_pred)
            direct_r2 = r2_score(subj_true, subj_pred) if len(subj_true) > 1 else 0
            direct_corr = np.corrcoef(subj_true, subj_pred)[0, 1] if len(subj_true) > 2 else 0
            print(f"  AI-PPG Age (zero-shot): MAE={direct_mae:.2f}, R2={direct_r2:.3f}, Corr={direct_corr:.3f}")
            print(f"    Pred range: {subj_pred.min():.1f} - {subj_pred.max():.1f}")
            print(f"    True range: {subj_true.min():.1f} - {subj_true.max():.1f}")

            # 2) Embeddings + Ridge (like other models)
            print("  Extracting embeddings...")
            emb_aippg_valid = extract_embeddings_aippg(aippg_model, device, beat_templates[beat_valid])
            print(f"  AI-PPG embeddings: {emb_aippg_valid.shape}")
            r = train_and_evaluate_kfold(emb_aippg_valid, beat_ages, beat_subjects, "AI-PPG Age")
            results.append(r)

            # Store full-size embedding array for fusion
            emb_aippg = np.zeros((len(signals), emb_aippg_valid.shape[1]), dtype=np.float32)
            emb_aippg[beat_valid] = emb_aippg_valid

        del aippg_model
    except Exception as e:
        print(f"AI-PPG Age failed: {e}")
        import traceback; traceback.print_exc()

    # PPG + demographics fusion
    print("\n--- PPG + Demographics fusion ---")
    demo_feats = np.column_stack([
        demographics['gender'], demographics['bmi'],
        demographics['height'], demographics['weight'],
        demographics['sbp'], demographics['dbp']
    ])
    valid = ~np.any(np.isnan(demo_feats), axis=1)
    for label, emb in [("PaPaGei-S+Demo", emb_papagei), ("Pulse-PPG+Demo", emb_pulseppg)]:
        if emb is None:
            continue
        fused = np.column_stack([emb[valid], demo_feats[valid]])
        r = train_and_evaluate_kfold(fused, ages[valid],
                                      [s for s, v in zip(subjects, valid) if v], label)
        results.append(r)

    # AI-PPG Age + Demographics fusion (only for valid beat segments)
    if emb_aippg is not None:
        valid_aippg = beat_valid & valid
        if valid_aippg.sum() > 100:
            fused = np.column_stack([emb_aippg[valid_aippg], demo_feats[valid_aippg]])
            r = train_and_evaluate_kfold(fused, ages[valid_aippg],
                                          [s for s, v in zip(subjects, valid_aippg) if v], "AI-PPG+Demo")
            results.append(r)

    # Save results
    if results:
        save_results(results, output_dir)

    # Learning curve
    if emb_papagei is not None or emb_pulseppg is not None:
        print("\n--- Computing learning curve ---")
        compute_learning_curve(emb_papagei, emb_pulseppg, ages, subjects, output_dir)


def compute_learning_curve(emb_papagei, emb_pulseppg, ages, subjects, output_dir):
    """Compute and plot MAE vs number of training subjects."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error

    unique_subjects = list(set(subjects))
    n_total = len(unique_subjects)
    subject_ids_arr = np.array(subjects)

    # Sample sizes to test
    sizes = sorted(set([20, 30, 50, 75, 100, 125, 150, 200, 250, 300]) & set(range(10, n_total)))
    # Always include current max (leave 20% for test)
    max_train = int(n_total * 0.8)
    sizes = [s for s in sizes if s <= max_train]
    if max_train not in sizes:
        sizes.append(max_train)
    sizes = sorted(sizes)

    n_repeats = 5
    results = {}

    for label, embeddings in [("PaPaGei-S", emb_papagei), ("Pulse-PPG", emb_pulseppg)]:
        if embeddings is None:
            continue
        curve = []
        for n_train_subj in sizes:
            maes = []
            for seed in range(n_repeats):
                rng = np.random.RandomState(seed)
                perm = rng.permutation(unique_subjects)
                train_subj = set(perm[:n_train_subj])
                test_subj = set(perm[n_train_subj:])

                train_mask = np.array([s in train_subj for s in subjects])
                test_mask = ~train_mask

                if test_mask.sum() == 0:
                    continue

                X_tr = embeddings[train_mask]
                y_tr = ages[train_mask]
                X_te = embeddings[test_mask]
                y_te = ages[test_mask]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)

                ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
                ridge.fit(X_tr_s, y_tr)
                y_pred = ridge.predict(X_te_s)

                # Subject-level MAE
                test_subj_list = [s for s, m in zip(subjects, test_mask) if m]
                sp, st = {}, {}
                for s, p, t in zip(test_subj_list, y_pred, y_te):
                    sp.setdefault(s, []).append(p)
                    st.setdefault(s, []).append(t)
                subj_pred = np.array([np.mean(sp[s]) for s in sp])
                subj_true = np.array([st[s][0] for s in st])
                maes.append(mean_absolute_error(subj_true, subj_pred))

            mean_mae = np.mean(maes)
            std_mae = np.std(maes)
            curve.append((n_train_subj, mean_mae, std_mae))
            print(f"  {label}: {n_train_subj} train subj -> MAE {mean_mae:.2f} +/- {std_mae:.2f}")

        results[label] = curve

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {'PaPaGei-S': '#2196F3', 'Pulse-PPG': '#FF9800'}
        for label, curve in results.items():
            ns = [c[0] for c in curve]
            means = [c[1] for c in curve]
            stds = [c[2] for c in curve]
            ax.plot(ns, means, 'o-', color=colors.get(label, 'gray'), label=label)
            ax.fill_between(ns,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.2, color=colors.get(label, 'gray'))

        # Reference lines
        baseline_mae = np.mean(np.abs(ages - np.mean(ages)))
        ax.axhline(y=baseline_mae, color='gray', linestyle='--', alpha=0.7, label=f'Baseline (predict mean): {baseline_mae:.1f}')
        ax.axhline(y=2.43, color='green', linestyle=':', alpha=0.7, label='Apple PpgAge (2.43, healthy, 213K)')

        ax.set_xlabel('Number of training subjects')
        ax.set_ylabel('Subject-level MAE (years)')
        ax.set_title('Learning Curve: Age Prediction from PPG Embeddings\n(PulseDB VitalDB, clinical finger PPG)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "learning_curve.png"
        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=150)
        print(f"\nLearning curve saved: {plot_path}")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping learning curve plot")


if __name__ == "__main__":
    main()
