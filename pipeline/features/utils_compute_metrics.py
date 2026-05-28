"""
Script with functions to compute HRV and PRV metrics.

Specifically, the script calculates HRV/PRV metrics (time, frequency and nonlinear domain) 
on sliding windows. 
It also plots the trends of HRV and PRV features across the different windows.
"""


import numpy as np
import neurokit2 as nk
import pandas as pd



# Metrics of interest for analysis
metrics_time = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD"]
titles_time = ["Mean RR", "SDNN", "RMSSD"]

#metrics_time = ["HRV_MeanNN", "HRV_SDNN"]
#titles_time = ["Mean NN", "SDNN"]

metrics_freq = ["HRV_LFn", "HRV_HFn", "HRV_LFHF", "HRV_VLF", "HRV_LF", "HRV_HF"]
titles_freq = ["LFn", "HFn", "LF/HF", "VLF", "LF", "HF"]

metrics_nonlinear = ["HRV_ApEn", "HRV_SampEn", "HRV_DFA_alpha1", "HRV_DFA_alpha2", "HRV_SD1", "HRV_SD2", "HRV_SD1SD2"]
titles_nonlinear = ["ApEn", "SampEn", "DFA α₁", "DFA α₂", "SD1", "SD2", "SD1/SD2"]

all_metrics = metrics_time + metrics_freq + metrics_nonlinear
all_titles = titles_time + titles_freq + titles_nonlinear


# =============================================================================
# COMPUTE HRV AND PRV FEATURES FUNCTION
# =============================================================================

def compute_metrics(hr, pr, time_hr, time_pr, window_sec, step_sec):
    """
    Compute HRV and PRV metrics.

    This function divides the HR/PR time series into overlapping windows, computes
    metrics (time, frequency and nonlinear domains) for both ECG and PPG and
    removes windows with NaN values.

    Parameters:
    - hr (array): Heart rate from ECG.
    - pr (array): Pulse rate from PPG.
    - time_hr (array): Time vector for HR.
    - time_pr (array): Time vector for PR.
    - window_sec (int): Length of the analysis window in seconds.
    - step_sec (int): Step size in seconds for sliding the analysis window.

    Returns:
    - df_hrv_cleaned (DataFrame): Cleaned DataFrame of HRV metrics from ECG (with NaN windows removed).
    - df_prv_cleaned (DataFrame): Cleaned DataFrame of PRV metrics from PPG (with NaN windows removed).
    """

    # Initialize lists to store HRV and PRV metrics for each modality and domain
    hrv_result_time, prv_result_time = [], []
    hrv_result_freq, prv_result_freq = [], []
    hrv_result_nonlinear, prv_result_nonlinear = [], []

    i = 0
    start_time = 0

    # Sliding window loop
    while start_time + window_sec <= min(time_hr[-1], time_pr[-1]):
        end_time = start_time + window_sec
        i += 1

        # Identify indices within the window based on time axis
        mask_ecg = (time_hr >= start_time) & (time_hr <= end_time)
        mask_ppg = (time_pr >= start_time) & (time_pr <= end_time)

        idx_ecg = np.where(mask_ecg)[0]
        idx_ppg = np.where(mask_ppg)[0]

        print(f"Window {i}: from {start_time:.0f}s to {end_time:.0f}s → {mask_ecg.sum()} samples HR and {mask_ppg.sum()} samples PR")

        # Skip window if not enough samples
        if len(idx_ecg) < 3 or len(idx_ppg) < 3:
            print(f"Window {i}: too few samples (HR: {len(idx_ecg)}, PR: {len(idx_ppg)}) → Skipping")
            start_time += step_sec
            continue

        # Extract HR and PR data in the selected window
        hr_window = hr[idx_ecg[0]:idx_ecg[-1]]
        pr_window = pr[idx_ppg[0]:idx_ppg[-1]]
        
        # Compute HRV and PRV metrics for each domain
        hrv_result_time.append(compute_metrics_per_window(hr_window, i))
        prv_result_time.append(compute_metrics_per_window(pr_window, i))

        hrv_result_freq.append(compute_metrics_per_window(hr_window, i, mode='freq'))
        prv_result_freq.append(compute_metrics_per_window(pr_window, i, mode='freq'))

        hrv_result_nonlinear.append(compute_metrics_per_window(hr_window, i, mode='nonlinear'))
        prv_result_nonlinear.append(compute_metrics_per_window(pr_window, i, mode='nonlinear'))

        start_time += step_sec

    # Concatenate metrics across all windows
    hrv_df_hrv_time = pd.concat(hrv_result_time, ignore_index=True)
    hrv_df_prv_time = pd.concat(prv_result_time, ignore_index=True)

    hrv_df_hrv_freq = pd.concat(hrv_result_freq, ignore_index=True)
    hrv_df_prv_freq = pd.concat(prv_result_freq, ignore_index=True)

    hrv_df_hrv_nonlinear = pd.concat(hrv_result_nonlinear, ignore_index=True)
    hrv_df_prv_nonlinear = pd.concat(prv_result_nonlinear, ignore_index=True)

    # Merge metrics from all domains for HRV and PRV
    df_hrv_all = (
        pd.merge(hrv_df_hrv_time[['Window'] + metrics_time], hrv_df_hrv_freq[['Window'] + metrics_freq], on='Window')
          .merge(hrv_df_hrv_nonlinear[['Window'] + metrics_nonlinear], on='Window')
    )

    df_prv_all = (
        pd.merge(hrv_df_prv_time[['Window'] + metrics_time], hrv_df_prv_freq[['Window'] + metrics_freq], on='Window')
          .merge(hrv_df_prv_nonlinear[['Window'] + metrics_nonlinear], on='Window')
    )

    # Remove windows containing NaN values
    df_hrv_cleaned, df_prv_cleaned = remove_windows_with_nan(df_hrv_all, df_prv_all)

    return df_hrv_cleaned, df_prv_cleaned


# =============================================================================
# Helper Functions
# =============================================================================

def compute_metrics_per_window(values, window_id, mode="time"):
    """
    Compute HRV/PRV metrics from heart/pulse rate values for a specific window
    using the appropriate NeuroKit functions.

    Parameters:
    - values (array): Heart/pulse rate values.
    - window_id (int): ID of the considered window.
    - mode (str): Analysis mode: 'time', 'freq', or 'nonlinear' domain. Default is 'time'.

    Returns:
    - df (DataFrame): HRV/PRV metrics for the specified mode and window.
    """
    
    # Convert HR/PR (in bpm) to RR intervals in milliseconds
    rr_intervals = (60 / values) * 1000  
    rr_dict = {"RRI": rr_intervals}
    
    # Compute metrics based on selected mode (Neurokit function)
    if mode == "freq":
        df = nk.hrv_frequency(rr_dict, sampling_rate=None)
    elif mode == "nonlinear":
        df = nk.hrv_nonlinear(rr_dict, sampling_rate=None)
    else:  # default to time domain
        df = nk.hrv_time(rr_dict, sampling_rate=None)

    # Add window identifier
    df["Window"] = f"Window_{window_id}"

    return df



def remove_windows_with_nan(df_hrv, df_prv):
    """
    Removes windows with NaN values in the specified HRV/PRV metrics from both dataframes.

    This function checks for missing values (NaN) in the given metrics for both HRV and PRV data across all windows.
    If a window contains NaN values for any of the specified metrics in either HRV or PRV, that window is removed
    from both dataframes.

    Parameters:
    - df_hrv (DataFrame): HRV metrics for ECG.
    - df_prv (DataFrame): PRV metrics for PPG.

    Returns:
    - df_hrv_clean (DataFrame): Cleaned dataframe for HRV data with windows containing NaN values removed.
    - df_prv_clean (DataFrame): Cleaned dataframe for PRV data with windows containing NaN values removed.
    """
    
    removed_windows = 0
    windows_to_remove = []
    
    for window_id in range(len(df_prv)):  
        # Check if there are NaN values in the metrics for the current window
        ecg_nan = df_hrv.loc[window_id, all_metrics].isna().any()
        ppg_nan = df_prv.loc[window_id, all_metrics].isna().any()
        
        if ppg_nan or ecg_nan:  
            # If there are NaN values, mark the window for removal
            windows_to_remove.append(window_id)
            print(f"Window {window_id} with NaN metric")
            removed_windows += 1  
            
            ecg_nan_metrics = df_hrv.loc[window_id, all_metrics][df_hrv.loc[window_id, all_metrics].isna()].index.tolist()
            ppg_nan_metrics = df_prv.loc[window_id, all_metrics][df_prv.loc[window_id, all_metrics].isna()].index.tolist()
            print(f"Metrics NaN in HRV (window {window_id}):", ecg_nan_metrics)
            print(f"Metrics NaN in PRV (window {window_id}):", ppg_nan_metrics)        
                   
    # Remove windows     
    df_hrv_clean = df_hrv.drop(windows_to_remove).reset_index(drop=True)
    df_prv_clean = df_prv.drop(windows_to_remove).reset_index(drop=True)
    print(f"\nWindows removed due to NaN in HRV or PRV data: {removed_windows}")
    
    return df_hrv_clean, df_prv_clean
