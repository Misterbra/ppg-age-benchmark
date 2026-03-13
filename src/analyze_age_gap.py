"""
WearAge Gap Analysis: Association between predicted age gap and cardiovascular health.

Computes age gap (predicted - chronological) from cross-validated predictions,
then tests correlation with SBP, DBP, BMI — replicating Apple PpgAge's key finding.
"""

import sys
import os
import numpy as np
import h5py
from pathlib import Path
from scipy.signal import resample
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "papagei-foundation-model"))
sys.path.insert(0, str(PROJECT_ROOT / "pulseppg"))

FS_PULSEDB = 125
SEGMENT_LEN_125 = 1250
SEGMENT_LEN_50 = 500
MIN_AGE = 18
MAX_SEGMENTS_PER_SUBJECT = 50


def load_data(data_dir):
    """Load PPG segments + demographics from VitalDB .mat files."""
    mat_dir = Path(data_dir)
    mat_files = sorted(mat_dir.glob("p*.mat"))

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
            ref = sw['Age'][0, 0]
            age = f[ref][0, 0]

            if age < MIN_AGE or np.isnan(age):
                f.close()
                continue

            gender_code = _read_float(f, sw, 'Gender', 0)
            bmi = _read_float(f, sw, 'BMI', 0)
            height = _read_float(f, sw, 'Height', 0)
            weight = _read_float(f, sw, 'Weight', 0)

            n_use = min(n_segs, MAX_SEGMENTS_PER_SUBJECT)
            indices = np.linspace(0, n_segs - 1, n_use, dtype=int)

            subject_id = fpath.stem
            for i in indices:
                ref = sw['PPG_F'][0, i]
                ppg = f[ref][:].flatten().astype(np.float32)
                if len(ppg) == SEGMENT_LEN_125:
                    all_signals.append(ppg)
                    all_ages.append(age)
                    all_subjects.append(subject_id)
                    all_demo['gender'].append(1.0 if gender_code == 77 else 0.0)
                    all_demo['bmi'].append(bmi)
                    all_demo['height'].append(height)
                    all_demo['weight'].append(weight)
                    all_demo['sbp'].append(_read_float(f, sw, 'SegSBP', i))
                    all_demo['dbp'].append(_read_float(f, sw, 'SegDBP', i))
            f.close()
        except Exception as e:
            pass

    signals = np.array(all_signals, dtype=np.float32)
    ages = np.array(all_ages, dtype=np.float32)
    demographics = {k: np.array(v, dtype=np.float32) for k, v in all_demo.items()}
    print(f"Loaded {len(ages)} segments from {len(set(all_subjects))} subjects")
    return signals, ages, all_subjects, demographics


def get_cv_predictions(embeddings, demo_feats, ages, subjects, n_folds=5):
    """Get cross-validated age predictions for every subject (using PPG+Demo fusion)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from collections import defaultdict

    # Build subject map
    subj_map = {}
    for i, (s, a) in enumerate(zip(subjects, ages)):
        subj_map.setdefault(s, {'indices': [], 'age': a})['indices'].append(i)

    unique_subjects = list(subj_map.keys())
    subj_ages = np.array([subj_map[s]['age'] for s in unique_subjects])

    # Stratify by decade
    decade_bins = (subj_ages // 10).astype(int)
    decade_groups = defaultdict(list)
    for i, d in enumerate(decade_bins):
        decade_groups[d].append(i)

    np.random.seed(42)
    fold_assignments = np.zeros(len(unique_subjects), dtype=int)
    for decade, indices in decade_groups.items():
        np.random.shuffle(indices)
        for j, idx in enumerate(indices):
            fold_assignments[idx] = j % n_folds

    # Fuse embeddings + demographics
    fused = np.column_stack([embeddings, demo_feats])

    # Collect predictions per subject
    subj_pred = {}
    subj_true = {}

    for fold in range(n_folds):
        test_subj = set(s for s, f in zip(unique_subjects, fold_assignments) if f == fold)
        train_subj = set(unique_subjects) - test_subj

        train_mask = np.array([s in train_subj for s in subjects])
        test_mask = ~train_mask

        X_train, y_train = fused[train_mask], ages[train_mask]
        X_test, y_test = fused[test_mask], ages[test_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
        ridge.fit(X_train_s, y_train)
        y_pred = ridge.predict(X_test_s)

        test_subj_list = [s for s, m in zip(subjects, test_mask) if m]
        for s, p, t in zip(test_subj_list, y_pred, y_test):
            subj_pred.setdefault(s, []).append(p)
            subj_true.setdefault(s, []).append(t)

    # Average predictions per subject
    result_subjects = list(subj_pred.keys())
    pred_ages = np.array([np.mean(subj_pred[s]) for s in result_subjects])
    true_ages = np.array([subj_true[s][0] for s in result_subjects])

    return result_subjects, true_ages, pred_ages


def get_subject_demographics(subjects_order, all_subjects, demographics):
    """Get subject-level demographics (mean across segments)."""
    from collections import defaultdict

    subj_demo = defaultdict(lambda: defaultdict(list))
    for i, s in enumerate(all_subjects):
        for key in demographics:
            subj_demo[s][key].append(demographics[key][i])

    result = {}
    for key in demographics:
        vals = []
        for s in subjects_order:
            v = np.nanmean(subj_demo[s][key])
            vals.append(v)
        result[key] = np.array(vals)
    return result


def analyze_gap(subjects, true_ages, pred_ages, subj_demo, output_dir):
    """Analyze age gap vs cardiovascular markers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    age_gap = pred_ages - true_ages
    n = len(subjects)

    print(f"\n{'='*60}")
    print(f"WEARIAGE GAP ANALYSIS ({n} subjects)")
    print(f"{'='*60}")
    print(f"  Age gap: mean={np.mean(age_gap):.2f}, std={np.std(age_gap):.2f}")
    print(f"  Range: [{np.min(age_gap):.1f}, {np.max(age_gap):.1f}]")

    # Correlations with health markers
    markers = {
        'SBP (mmHg)': subj_demo['sbp'],
        'DBP (mmHg)': subj_demo['dbp'],
        'BMI (kg/m²)': subj_demo['bmi'],
    }

    print(f"\n  {'Marker':<20} {'r':>8} {'p-value':>12} {'n':>5}")
    print(f"  {'-'*50}")

    results = {}
    for name, values in markers.items():
        valid = ~np.isnan(values) & ~np.isnan(age_gap)
        if valid.sum() < 10:
            continue
        r, p = stats.pearsonr(age_gap[valid], values[valid])
        # Also partial correlation controlling for age
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(true_ages[valid].reshape(-1, 1), age_gap[valid])
        gap_resid = age_gap[valid] - lr.predict(true_ages[valid].reshape(-1, 1))
        lr.fit(true_ages[valid].reshape(-1, 1), values[valid])
        marker_resid = values[valid] - lr.predict(true_ages[valid].reshape(-1, 1))
        r_partial, p_partial = stats.pearsonr(gap_resid, marker_resid)

        print(f"  {name:<20} {r:>8.3f} {p:>12.2e} {valid.sum():>5}")
        print(f"    (age-adjusted)   {r_partial:>8.3f} {p_partial:>12.2e}")
        results[name] = {'r': r, 'p': p, 'r_partial': r_partial, 'p_partial': p_partial,
                         'values': values, 'valid': valid}

    # Gender differences
    male = subj_demo['gender'] == 1.0
    female = subj_demo['gender'] == 0.0
    if male.sum() > 5 and female.sum() > 5:
        print(f"\n  Gender differences:")
        print(f"    Male   (n={male.sum()}): age gap = {np.mean(age_gap[male]):+.2f} +/- {np.std(age_gap[male]):.2f}")
        print(f"    Female (n={female.sum()}): age gap = {np.mean(age_gap[female]):+.2f} +/- {np.std(age_gap[female]):.2f}")
        t_stat, p_val = stats.ttest_ind(age_gap[male], age_gap[female])
        print(f"    t-test: t={t_stat:.2f}, p={p_val:.3f}")

    # Age gap by decade
    print(f"\n  Age gap by decade:")
    decades = [(20, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]
    for low, high in decades:
        mask = (true_ages >= low) & (true_ages < high)
        if mask.sum() > 0:
            print(f"    {low}-{high}: gap={np.mean(age_gap[mask]):+.1f} +/- {np.std(age_gap[mask]):.1f} (n={mask.sum()})")

    # Tertile analysis: compare health markers between "young-looking" and "old-looking"
    print(f"\n  Tertile analysis (bottom vs top third by age gap):")
    sorted_idx = np.argsort(age_gap)
    n_tertile = n // 3
    young_idx = sorted_idx[:n_tertile]  # Most negative gap = "younger than age"
    old_idx = sorted_idx[-n_tertile:]   # Most positive gap = "older than age"

    print(f"    Young-looking (n={len(young_idx)}): gap = {np.mean(age_gap[young_idx]):+.1f}yr")
    print(f"    Old-looking   (n={len(old_idx)}):  gap = {np.mean(age_gap[old_idx]):+.1f}yr")

    for name, values in markers.items():
        valid_y = ~np.isnan(values[young_idx])
        valid_o = ~np.isnan(values[old_idx])
        if valid_y.sum() > 5 and valid_o.sum() > 5:
            mean_y = np.nanmean(values[young_idx])
            mean_o = np.nanmean(values[old_idx])
            t, p = stats.ttest_ind(values[young_idx][valid_y], values[old_idx][valid_o])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    {name:<20}: young={mean_y:.1f}, old={mean_o:.1f}, diff={mean_o-mean_y:+.1f}, p={p:.3f} {sig}")

    # ── Plots ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Scatter: true vs predicted age
    ax = axes[0, 0]
    sc = ax.scatter(true_ages, pred_ages, c=age_gap, cmap='RdYlBu_r', alpha=0.7, s=30, vmin=-15, vmax=15)
    lims = [min(true_ages.min(), pred_ages.min()) - 5, max(true_ages.max(), pred_ages.max()) + 5]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('Chronological Age')
    ax.set_ylabel('Predicted Age (Pulse-PPG+Demo)')
    ax.set_title(f'Age Prediction (n={n})')
    plt.colorbar(sc, ax=ax, label='Age Gap (yr)')

    # 2-4. Gap vs health markers
    for idx, (name, info) in enumerate(results.items()):
        ax = axes[0, 1 + idx] if idx < 2 else axes[1, 0]
        valid = info['valid']
        ax.scatter(age_gap[valid], info['values'][valid], alpha=0.5, s=20, color='steelblue')
        # Regression line
        z = np.polyfit(age_gap[valid], info['values'][valid], 1)
        x_line = np.linspace(age_gap[valid].min(), age_gap[valid].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'r-', lw=2, alpha=0.7)
        ax.set_xlabel('Age Gap (predicted - actual)')
        ax.set_ylabel(name)
        r_raw = f"r={info['r']:.3f}, p={info['p']:.2f} (raw)"
        sig = info['p_partial'] < 0.05
        r_partial_str = f"r={info['r_partial']:.3f}, p={info['p_partial']:.1e} (age-adjusted)" if sig else f"r={info['r_partial']:.3f}, p={info['p_partial']:.2f} (age-adjusted)"
        ax.set_title(f"Age Gap vs {name}\n{r_raw}\n{r_partial_str}", fontsize=9)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    # 5. Age gap distribution
    ax = axes[1, 1]
    ax.hist(age_gap, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', lw=1.5)
    ax.set_xlabel('Age Gap (years)')
    ax.set_ylabel('Count')
    ax.set_title(f'Age Gap Distribution\nmean={np.mean(age_gap):.1f}, std={np.std(age_gap):.1f}')

    # 6. Tertile comparison (bar chart)
    ax = axes[1, 2]
    marker_names = []
    young_means = []
    old_means = []
    for name, values in markers.items():
        valid_y = ~np.isnan(values[young_idx])
        valid_o = ~np.isnan(values[old_idx])
        if valid_y.sum() > 5 and valid_o.sum() > 5:
            marker_names.append(name.split(' (')[0])
            young_means.append(np.nanmean(values[young_idx]))
            old_means.append(np.nanmean(values[old_idx]))
    x = np.arange(len(marker_names))
    w = 0.35
    ax.bar(x - w/2, young_means, w, label=f'Young-looking (gap<{age_gap[young_idx[-1]]:.0f}yr)', color='#2196F3')
    ax.bar(x + w/2, old_means, w, label=f'Old-looking (gap>{age_gap[old_idx[0]]:.0f}yr)', color='#FF5722')
    ax.set_xticks(x)
    ax.set_xticklabels(marker_names)
    ax.legend(fontsize=7)
    ax.set_title('Health Markers by Age Gap Tertile')
    ax.set_ylabel('Value')

    plt.suptitle('WearAge Gap: Association with Cardiovascular Health\n(PulseDB VitalDB, Pulse-PPG+Demographics)', fontsize=13)
    plt.tight_layout()
    plot_path = output_dir / "weariage_gap_analysis.png"
    plt.savefig(str(plot_path), dpi=150)
    print(f"\nPlot saved: {plot_path}")
    plt.close()


def main():
    import torch

    data_dir = PROJECT_ROOT / "data" / "pulsedb_segments" / "PulseDB_Vital"
    output_dir = PROJECT_ROOT / "results" / "pulsedb_vitaldb"

    print("Loading data...")
    signals, ages, subjects, demographics = load_data(data_dir)

    # Load Pulse-PPG model and extract embeddings
    print("\nExtracting Pulse-PPG embeddings...")
    from pulseppg.nets.ResNet1D.ResNet1D_Net import Net as ResNet1D
    model = ResNet1D(
        in_channels=1, base_filters=128, kernel_size=11,
        stride=2, groups=1, n_block=12,
        downsample_gap=2, increasefilter_gap=4, use_do=False, finalpool="max"
    )
    weights_path = PROJECT_ROOT / "pulseppg" / "pulseppg" / "experiments" / "out" / "pulseppg" / "checkpoint_best.pkl"
    checkpoint = torch.load(str(weights_path), map_location='cpu', weights_only=False)
    cleaned = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
    model.load_state_dict(cleaned)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Resample to 50Hz and extract embeddings
    signals_50 = np.array([resample(s, SEGMENT_LEN_50) for s in signals], dtype=np.float32)
    embeddings = []
    for i in range(0, len(signals_50), 32):
        batch = signals_50[i:i+32]
        batch = (batch - batch.mean(axis=1, keepdims=True)) / (batch.std(axis=1, keepdims=True) + 1e-8)
        tensor = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.inference_mode():
            emb = model(tensor).cpu().numpy()
        embeddings.append(emb)
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Embeddings: {embeddings.shape}")
    del model

    # Build demo features
    demo_feats = np.column_stack([
        demographics['gender'], demographics['bmi'],
        demographics['height'], demographics['weight'],
        demographics['sbp'], demographics['dbp']
    ])
    valid = ~np.any(np.isnan(demo_feats), axis=1)
    print(f"Valid samples (no NaN in demographics): {valid.sum()}/{len(valid)}")

    embeddings_v = embeddings[valid]
    ages_v = ages[valid]
    subjects_v = [s for s, v in zip(subjects, valid) if v]
    demo_feats_v = demo_feats[valid]

    # Get cross-validated predictions
    print("\nRunning 5-fold CV for age gap predictions...")
    result_subjects, true_ages, pred_ages = get_cv_predictions(
        embeddings_v, demo_feats_v, ages_v, subjects_v
    )

    # Get subject-level demographics
    subj_demo = get_subject_demographics(result_subjects, subjects_v,
                                          {k: demographics[k][valid] for k in demographics})

    # Run analysis
    analyze_gap(result_subjects, true_ages, pred_ages, subj_demo, output_dir)


if __name__ == "__main__":
    main()
