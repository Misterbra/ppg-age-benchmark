# Benchmarking Open-Source PPG Foundation Models for Biological Age Prediction

Code and results for the preprint (2026). Work in progress.

## What this is

Apple showed in October 2025 (Nature Communications) that PPG signals from a wrist sensor can predict biological age with 2.4-year MAE, and that this "PPG age gap" is associated with cardiovascular disease, diabetes, and mortality. A second group confirmed similar results on the UK Biobank (212K subjects). Both models are closed-source.

This repo benchmarks two open-source PPG foundation models (Pulse-PPG and PaPaGei-S) on the same task, using public clinical data (PulseDB/VitalDB, 906 subjects). No fine-tuning, just linear probing on frozen embeddings.

## Results (906 VitalDB subjects, 5-fold stratified CV)

| Model | MAE (years) | R2 | Corr |
|---|---|---|---|
| Baseline (predict mean) | 11.91 | | |
| HR only | 11.83 | | |
| HR + HRV | 11.49 | | |
| Demographics (all) | 10.04 | | |
| HR/HRV + Demographics | 9.59 | | |
| AI-PPG Age (zero-shot) | 11.50 | 0.043 | 0.313 |
| PaPaGei-S (linear probe) | 9.80 +/- 0.34 | 0.313 | 0.583 |
| AI-PPG Age (linear probe) | 9.72 +/- 0.37 | 0.310 | 0.563 |
| Pulse-PPG (linear probe) | 9.28 +/- 0.44 | 0.388 | 0.645 |
| PaPaGei-S + Demographics | 8.56 +/- 0.22 | 0.477 | 0.696 |
| AI-PPG Age + Demographics | 8.58 +/- 0.25 | 0.440 | 0.664 |
| **Pulse-PPG + Demographics** | **8.22 +/- 0.25** | **0.517** | **0.725** |

Three things stand out. Pulse-PPG embeddings alone (MAE 9.28) beat HR/HRV combined with all demographics (9.59), so the waveform morphology carries real aging information beyond heart rate. AI-PPG Age fails zero-shot on clinical data (predictions stuck at 38-67yr for a population aged 18-92yr), which illustrates the domain shift problem. And fusion with demographics pushes the best result to 8.22 years.

The age-adjusted PPG age gap correlates with diastolic blood pressure (r=-0.188, p=1.2e-8), consistent with Apple's finding that PPG captures vascular aging.

## Project structure

```
PPGAge/
├── src/
│   ├── run_pulsedb_age.py     # Main benchmark pipeline (5-fold CV)
│   ├── analyze_age_gap.py     # PPG age gap vs cardiovascular markers
│   ├── run_ppgbp_age.py       # PPG-BP dataset analysis
│   └── analyze_pulsedb_demographics.py
├── data/
│   ├── vitaldb_fileids.json   # All 2938 VitalDB Google Drive file IDs
│   └── download_list.json     # Selected subjects to download
├── results/
│   └── pulsedb_vitaldb/       # Plots and full output log
├── pulseppg/                  # Pulse-PPG (clone separately, MIT license)
├── papagei-foundation-model/  # PaPaGei (clone separately)
├── ppg-vascularage/           # AI-PPG Age weights + patched net1d.py
└── download_vitaldb.py        # Script to download VitalDB segments
```

## Resources

| Resource | Link |
|---|---|
| PpgAge paper (Apple, Nature Comms 2025) | [doi](https://www.nature.com/articles/s41467-025-64275-4) |
| AI-PPG Age paper (Comms Medicine 2025) | [doi](https://www.nature.com/articles/s43856-025-01188-9) |
| Pulse-PPG (MIT) | [github.com/maxxu05/pulseppg](https://github.com/maxxu05/pulseppg) |
| PaPaGei | [github.com/Nokia-Bell-Labs/papagei-foundation-model](https://github.com/Nokia-Bell-Labs/papagei-foundation-model) |
| AI-PPG Age weights | [HuggingFace Ngks03/PPG-VascularAge](https://huggingface.co/Ngks03/PPG-VascularAge) |
| PulseDB dataset | [github.com/pulselabteam/PulseDB](https://github.com/pulselabteam/PulseDB) |

## Setup

```bash
git clone https://github.com/Misterbra/ppg-age-benchmark
cd ppg-age-benchmark
pip install -r requirements.txt

# Clone the model repos
git clone https://github.com/maxxu05/pulseppg
git clone https://github.com/Nokia-Bell-Labs/papagei-foundation-model

# Download weights
cd pulseppg && bash download_model.sh && cd ..
# For PaPaGei: see papagei-foundation-model/README.md
# For AI-PPG Age: huggingface-cli download Ngks03/PPG-VascularAge --local-dir ppg-vascularage/

# Download VitalDB segments (needs Google Drive access, see script)
python download_vitaldb.py
```

## Run

```bash
python src/run_pulsedb_age.py
# Results saved to results/pulsedb_vitaldb/
```

## Citation

```bibtex
@article{brag2026ppgage,
  title={Benchmarking Open-Source PPG Foundation Models for Biological Age Prediction},
  author={Brag, N.},
  year={2026},
  note={Preprint}
}
```
