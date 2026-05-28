# 💓 HRV/PRV Comparative Analysise

This project provides a complete pipeline for analyzing cardiac variability derived from ECG signals (Heart Rate Variability – HRV) and PPG signals (Pulse Rate Variability – PRV).


The pipeline includes acquisition of ECG and PPG signals recorded during sleep, proper preprocessing steps and identification of R-peaks in ECG and systolic peaks in PPG.
Noisy segments, detected using a custom-developed algorithm, are removed from the signals, and HRV and PRV metrics are then computed in time, frequency and non linear domains.
Comparative analyses are performed to assess the similarity between metrics.

<div align="center">
  <img src="https://github.com/user-attachments/assets/abb63f0c-83f0-4202-8add-a5412ff58cfe" width="800"/>
</div>

The workflow starts from `.edf` files (polysomnographic recordings) and `.txt` files (hypnograms) of the subjects.
These files are processed to extract peaks from the ECG and PPG signals (saved as `.npz` files).
Based on R-peaks and systolic peaks, Heart Rate (HR) and Pulse Rate (PR) are calculated.
Noisy windows are removed from HR and PR. Subsequently, HRV and PRV metrics are computed over 5-minute windows with 30% overlap (saved as `.csv` files), both for the entire signals and after excluding noisy segments.
Finally, several comparative analyses are performed between HRV and PRV metrics, including Pearson and Spearman correlations, Bland-Altman plots, and TOST equivalence tests.

## Repository Content and Structure

```text
HRV-PRV-Comparison/
├── README.md
├── LICENSE
├── requirements.txt
│
├── Data/
│   ├── All_Windows/
│   │   ├── HRV/
│   │   │   └── S#.csv        # HRV features (no artifact removal)
│   │   └── PRV/
│   │       └── S#.csv        # PRV features (no artifact removal)
│   │
│   └── Window_Removal/
│       ├── HRV/
│       │   └── S#.csv        # HRV features (M2 artifact removal)
│       └── PRV/
│           └── S#.csv        # PRV features (M2 artifact removal)
│
├── pipeline/
│   ├── preprocessing/        # ECG/PPG preprocessing + peak detection (Step 01)
│   │   ├── peaks_identification.py
│   │   ├── utils_peaks_correction.py
│   │   └── utils_signal_processing.py
│   │
│   ├── features/             # HRV/PRV feature extraction (Step 02)
│   │   ├── compute_features.ipynb
│   │   ├── utils_compute_metrics.py
│   │   ├── utils_remove_windows.py
│   │   └── window_removal_visualization.ipynb
│   │
│   └── analysis/             # Comparative analysis pipeline (Step 03)
│       ├── analysis.ipynb
│       ├── analysis_correlations_per_group.ipynb
│       ├── equivalence_test.ipynb
│       ├── plot_feature_corr_per_subj.ipynb
│       ├── statistical_test.ipynb
│       ├── utils_analysis.py
│       ├── utils_analysis_plot.py
│       └── utils_statistical_tests.py
│
├── results/                  # Correlation / statistical / equivalence outputs
└── figures/                  # Generated figures

 ```  
 
## 📁 Files Description

- `pipeline/`: Contains scripts to run the full analysis pipeline.
- `Data/`: Contains HRV and PRV features for all subjects. This folder has two subfolders: `All_Windows` and `Window_Removal`***. 
Each subfolder contains `HRV` and `PRV` folders with `.csv` files for all subjects. In this project, a dataset of 50 subjects was used, divided into five equally sized groups (10 subjects each, in order):
	- Healthy
	- RBD (REM Sleep Behavior Disorder)
 	- OSAS (Obstructive Sleep Apnea Syndrome)
	- PLM (Periodic Limb Movements)
 	- Subjects with multiple comorbidities.  

***This folder has two subfolders: All Windows and M2.


## 🔄 Pipeline Overview 

The scripts are organized into three separate folders, each corresponding to a specific step of the pipeline.

<img width="2060" height="458" alt="Screenshot 2025-10-29 100535" src="https://github.com/user-attachments/assets/ef6a5f79-825e-49fc-b53e-dd5b092c42ee" />

### 🧩 Step 01 — Signal Processing & Peak Detection
- `peaks_identification.py`: Processing of ECG and PPG signals, and ECG R-peaks and PPG systolic peaks detection for a single subject.
	The data used in this work were acquired with two different devices (Nox and Somté) and organized in different formats; therefore, the pipeline is designed to handle both.  
  *Inputs:* `.edf` file (PSG), `.txt` file (hypnograms)  
  *Outputs:* `.npz` file with processed signals and peaks  

	The file paths need to be specified.


### ⚙️ Step 02 — Feature Extraction
- `window_removal_visualization.py`: Removal of windows identified as noisy and generation of useful visualizations.   
  *Inputs:* Peak files (`.npz`)  
  *Outputs:* Figures
- `compute_features.py`: Computes HRV and PRV features on 5-minute windows (30% overlap) with and without window removal.   
  *Inputs:* Peak files (`.npz`)  
  *Outputs:* `.csv` files for HRV and PRV features   

The file paths in these scripts need to be specified.

### 📊 Step 03 — Comparative Analysis
- `analysis.py`: Global HRV and PRV metric analysis across all 50 subjects.  
Specifically, the analysis includes visualizing HRV and PRV metrics using violin plots, computing Pearson and Spearman correlations which are represented through both bar plots and boxplots, generating heatmaps of the coefficient of variation (CV) and interquartile range (IQR) values, and creating Bland-Altman plots for each metric to assess agreement.
- `analysis_correlations_per_group.py`: Global correlations by subject group (HC, RBD, OSAS, PLM, mixed).
- `plot_feature_corr_per_subj.py`: Plot correlations and metrics trend for each subject. 
- `equivalence_test.py`: Equivalence tests on HRV and PRV features for each individual subject.  
- `statistical_test.py`: Statistical tests on HRV and PRV features for each individual subject.    

The data in these scripts are taken directly from the `.csv` files in the `Data` folder.


## Data availability
This repository contains processed and anonymized data derived from polysomnographic recordings, including extracted features and statistical analysis outputs.
No raw clinical recordings are shared. The processed data released in this repository are distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

