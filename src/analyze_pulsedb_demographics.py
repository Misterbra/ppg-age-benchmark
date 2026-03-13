"""
Analyze PulseDB demographics: age distribution, sex ratio, MIMIC vs VitalDB split.
Run this after downloading PulseDB_Info.mat.
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def analyze_pulsedb_info(info_path):
    """Load PulseDB_Info.mat and analyze demographics."""
    import scipy.io as sio

    print(f"Loading {info_path}...")
    try:
        mat = sio.loadmat(str(info_path))
    except NotImplementedError:
        # v7.3 mat files need h5py
        import h5py
        print("  -> v7.3 .mat file, using h5py...")
        with h5py.File(str(info_path), 'r') as f:
            print(f"  Keys: {list(f.keys())}")
            analyze_h5py(f)
            return

    print(f"Keys in .mat file: {list(mat.keys())}")

    # Find the data structure
    for key in mat.keys():
        if key.startswith('_'):
            continue
        data = mat[key]
        print(f"\n  Key '{key}': type={type(data)}, shape={getattr(data, 'shape', 'N/A')}")
        if hasattr(data, 'dtype') and data.dtype.names:
            print(f"  Fields: {data.dtype.names}")

    # Try to extract age, gender, source info
    # PulseDB Info structure has fields: Subj_Name, Seg_Idx, Age, Gender, etc.
    info = None
    for key in mat.keys():
        if not key.startswith('_'):
            info = mat[key]
            break

    if info is None:
        print("Could not find data in .mat file")
        return

    extract_demographics(info)


def deref_float(f, ds, key):
    """Dereference h5py object references to float array."""
    n = ds[key].shape[1]
    vals = np.zeros(n)
    for i in range(n):
        ref = ds[key][0, i]
        val = f[ref][0, 0]
        vals[i] = val
    return vals


def deref_str(f, ds, key):
    """Dereference h5py object references to string list."""
    n = ds[key].shape[1]
    strs = []
    for i in range(n):
        ref = ds[key][0, i]
        chars = f[ref][:].flatten()
        s = ''.join(chr(int(c)) for c in chars)
        strs.append(s)
    return strs


def analyze_h5py(f):
    """Analyze demographics from h5py file (v7.3 .mat)."""
    ds = f['Dataset_Info']
    n = ds['Subj_Age'].shape[1]
    print(f"\n  Total subjects: {n}")

    # Load ages
    print("  Loading ages...")
    ages = deref_float(f, ds, 'Subj_Age')
    valid = ages > 0
    ages = ages[valid]
    analyze_ages(ages)

    # Load genders (char codes: F=70, M=77)
    print("  Loading genders...")
    genders_raw = deref_float(f, ds, 'Subj_Gender')
    genders = []
    for g in genders_raw:
        genders.append(chr(int(g)) if g > 0 else '?')
    analyze_genders_chars(genders)

    # Load sources
    print("  Loading sources...")
    sources = deref_str(f, ds, 'Source')
    analyze_sources_direct(sources)


def extract_demographics(info):
    """Extract and analyze demographics from loaded info structure."""
    # Handle structured numpy array
    if hasattr(info, 'dtype') and info.dtype.names:
        names = info.dtype.names
        print(f"\nFields available: {names}")

        if 'Age' in names:
            ages = info['Age'].flatten().astype(float)
            # Filter out NaN/invalid
            ages = ages[~np.isnan(ages)]
            analyze_ages(ages)

        if 'Gender' in names:
            genders = info['Gender'].flatten()
            analyze_genders(genders)

        if 'Subj_Name' in names:
            subjects = info['Subj_Name'].flatten()
            analyze_sources(subjects, info)
    else:
        print(f"Info type: {type(info)}, shape: {getattr(info, 'shape', 'N/A')}")
        print("Trying to parse as cell array / struct array...")
        print(repr(info)[:500])


def analyze_ages(ages):
    """Analyze age distribution."""
    print(f"\n{'='*60}")
    print("AGE DISTRIBUTION")
    print(f"{'='*60}")
    print(f"  N subjects/segments: {len(ages)}")
    print(f"  Mean age:   {np.mean(ages):.1f} years")
    print(f"  Median age: {np.median(ages):.1f} years")
    print(f"  Std dev:    {np.std(ages):.1f} years")
    print(f"  Min:        {np.min(ages):.0f} years")
    print(f"  Max:        {np.max(ages):.0f} years")

    # Distribution by decade
    print(f"\n  Age distribution by decade:")
    decades = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
    for low, high in decades:
        count = np.sum((ages >= low) & (ages < high))
        pct = count / len(ages) * 100
        bar = '#' * int(pct / 2)
        print(f"    {low:3d}-{high:3d}: {count:6d} ({pct:5.1f}%) {bar}")

    # Baseline MAE (predict mean)
    baseline_mae = np.mean(np.abs(ages - np.mean(ages)))
    print(f"\n  Baseline MAE (predict mean): {baseline_mae:.2f} years")
    print(f"  -> This is the floor. Any model must beat this.")

    # Quartiles
    q25, q50, q75 = np.percentile(ages, [25, 50, 75])
    print(f"\n  Quartiles: Q25={q25:.0f}, Q50={q50:.0f}, Q75={q75:.0f}")
    print(f"  IQR: {q75-q25:.0f} years")


def analyze_genders(genders):
    """Analyze gender distribution."""
    print(f"\n{'='*60}")
    print("GENDER DISTRIBUTION")
    print(f"{'='*60}")
    try:
        # Try to decode if bytes
        decoded = []
        for g in genders:
            if isinstance(g, (bytes, np.bytes_)):
                decoded.append(g.decode())
            elif isinstance(g, np.ndarray):
                decoded.append(str(g.item()) if g.size == 1 else str(g))
            else:
                decoded.append(str(g))
        from collections import Counter
        counts = Counter(decoded)
        total = sum(counts.values())
        for gender, count in counts.most_common():
            print(f"  {gender}: {count} ({count/total*100:.1f}%)")
    except Exception as e:
        print(f"  Could not parse genders: {e}")
        print(f"  Raw sample: {genders[:5]}")


def analyze_sources(subjects, info):
    """Analyze MIMIC vs VitalDB split based on subject names."""
    print(f"\n{'='*60}")
    print("DATA SOURCE SPLIT (MIMIC vs VitalDB)")
    print(f"{'='*60}")
    try:
        mimic_count = 0
        vital_count = 0
        other_count = 0
        for s in subjects:
            name = str(s).lower()
            if 'mimic' in name or name.startswith('p'):  # MIMIC subjects often start with 'p'
                mimic_count += 1
            elif 'vital' in name or name.startswith('v'):
                vital_count += 1
            else:
                other_count += 1
        total = mimic_count + vital_count + other_count
        print(f"  MIMIC-III:  ~{mimic_count} ({mimic_count/total*100:.1f}%)")
        print(f"  VitalDB:   ~{vital_count} ({vital_count/total*100:.1f}%)")
        if other_count:
            print(f"  Other:     ~{other_count} ({other_count/total*100:.1f}%)")
    except Exception as e:
        print(f"  Could not parse sources: {e}")


def analyze_genders_chars(genders):
    """Analyze gender from char list (F/M)."""
    print(f"\n{'='*60}")
    print("GENDER DISTRIBUTION")
    print(f"{'='*60}")
    from collections import Counter
    counts = Counter(genders)
    total = len(genders)
    label_map = {'M': 'Male', 'F': 'Female'}
    for val, count in counts.most_common():
        label = label_map.get(val, f'Unknown ({val})')
        print(f"  {label}: {count} ({count/total*100:.1f}%)")
    print(f"  Total: {total}")


def analyze_sources_direct(sources):
    """Analyze MIMIC vs VitalDB split from string list."""
    print(f"\n{'='*60}")
    print("DATA SOURCE SPLIT (MIMIC vs VitalDB)")
    print(f"{'='*60}")
    from collections import Counter
    counts = Counter(sources)
    total = len(sources)
    for val, count in counts.most_common():
        print(f"  {val}: {count} ({count/total*100:.1f}%)")


def main():
    # Check multiple possible locations
    paths = [
        PROJECT_ROOT / "data" / "pulsedb_info" / "PulseDB_Info.mat",
        PROJECT_ROOT / "PulseDB" / "Info_Files" / "PulseDB_Info.mat",
        PROJECT_ROOT / "data" / "PulseDB_Info.mat",
    ]

    info_path = None
    for p in paths:
        if p.exists():
            info_path = p
            break

    if info_path is None:
        print("PulseDB_Info.mat not found. Checked:")
        for p in paths:
            print(f"  {p}")
        print("\nDownload from: https://drive.google.com/drive/folders/1-0IAdIHT9AXpQ8I1Z-xS5PqB3Yeo-bNY")
        return

    analyze_pulsedb_info(info_path)


if __name__ == "__main__":
    main()
