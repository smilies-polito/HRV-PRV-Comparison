"""
This code provides functions for removing HR/PR windows based on the detection of outliers 
(values outside mean ± 3*SD) and the number of peaks in |HR - PR|. 
It also includes functions for visualizing the signals with the removed areas highlighted.
"""


import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import find_peaks
from scipy.signal import correlate


# =============================================================================
# HR PR ALIGNMENT FUNCTION
# =============================================================================

def align_pr_to_hr(ecg_peaks_final, ppg_peaks_final, fs_ecg, fs_ppg, len_ecg):
    """
    Align PPG-derived Pulse Rate (PR) signal to ECG-derived Heart Rate (HR) 
    by estimating and correcting the temporal lag between the two signals 
    using cross-correlation.

    Parameters:
    - ecg_peaks_final (array): Detected ECG peak indices.
    - ppg_peaks_final (array): Detected PPG peak indices.
    - fs_ecg (int or float): ECG sampling frequency.
    - fs_ppg (int or float): PPG sampling frequency.
    - len_ecg (int): Length of the ECG signal (used to match HR and PR lengths).

    Returns:
    - hr_final (array): ECG-derived Heart Rate (in bpm) after alignment.
    - pr_final (array): PPG-derived Pulse Rate (in bpm) after alignment.
    - time_hr (array): Time vector for HR.
    - time_pr (array): Time vector for PR.
    - time_shift (float): Estimated lag (in seconds) between PR and HR.
    """

    # 1) Compute HR and PR with the same length of ecg (for cross-correlation)
    hr = nk.ecg_rate(ecg_peaks_final, sampling_rate=fs_ecg, desired_length=len_ecg)
    pr = nk.ppg_rate(ppg_peaks_final, sampling_rate=fs_ppg, desired_length=len_ecg)

    # 2) Compute cross-correlation to find temporal lag
    hr_demeaned = hr - np.mean(hr)
    pr_demeaned = pr - np.mean(pr)
    correlation = correlate(pr_demeaned, hr_demeaned, mode='full')
    lags = np.arange(-len(hr_demeaned) + 1, len(hr_demeaned))

    # Find lag (in samples) corresponding to max correlation
    lag = lags[np.argmax(correlation)]
    time_shift = lag / fs_ecg  # convert to seconds
    print(f"Lag: {lag} samples ({time_shift:.3f} s)")

    if abs(time_shift) > 1:
        lag = 0.2 * fs_ecg
        print(f"Time shift > 1s → applying a delay of {lag} samples (≈200 ms)")

    # 3) Shift PPG peaks by the estimated lag
    ppg_peaks_aligned = ppg_peaks_final - lag

    # 4) Recompute PR and HR after alignment (with real lenght)
    hr_final = nk.ecg_rate(ecg_peaks_final, sampling_rate=fs_ecg)
    pr_final = nk.ppg_rate(ppg_peaks_aligned, sampling_rate=fs_ppg)

    # Remove the first element (automatically added by NeuroKit)
    hr_final = hr_final[1:]
    pr_final = pr_final[1:]

    # 5) Create corresponding time vectors (assigning HR to the first value of the interval)
    time_hr = ecg_peaks_final[:-1] / fs_ecg
    time_pr = ppg_peaks_aligned[:-1] / fs_ppg

    return hr_final, pr_final, time_hr, time_pr, time_shift


# =============================================================================
# PARAMETERS FUNCTION
# =============================================================================

def compute_peaks_difference(hr_final, pr_final, time_hr, time_pr, h_min):
    """
    Compute the absolute difference between ECG-derived Heart Rate (HR) 
    and PPG-derived Pulse Rate (PR), identify local peaks in this difference 
    and return their locations.

    Parameters:
    - hr_final (array): ECG-derived Heart Rate (in bpm).
    - pr_final (array): PPG-derived Pulse Rate (in bpm).
    - time_hr (array): Time vector for HR.
    - time_pr (array): Time vector for PR.
    - h_min (float): Minimum peak height threshold to identify relevant |HR - PR| peaks (in bpm).

    Returns:
    - time_common (array): Common interpolated time vector (seconds).
    - diff (array): Absolute difference |HR - PR| over time.
    - diff_peaks (array): Indices of peaks in `diff` that exceed `h_min`.
    """

    # HR and PR interpolation (in order to make a difference)
    start_time = max(time_hr[0], time_pr[0])
    end_time = min(time_hr[-1], time_pr[-1]) 
    time_common = np.arange(start_time, end_time, 0.5)

    hr_interp = np.interp(time_common, time_hr, hr_final)
    pr_interp = np.interp(time_common, time_pr, pr_final)

    # Absolute value of the HR-PR and peaks
    diff = abs(hr_interp - pr_interp)
    diff_peaks, _ = find_peaks(diff, height=h_min)

    return time_common, diff, diff_peaks



def compute_mean_std(hr_final, pr_final):
    """
    Compute mean and standard deviation (±3*std) bounds for ECG-derived Heart Rate (HR) 
    and PPG-derived Pulse Rate (PR) signals.  
    These bounds are used for detecting outliers in subsequent signal cleaning steps.

    Parameters:
    - hr_final (array): ECG-derived Heart Rate (in bpm).
    - pr_final (array): PPG-derived Pulse Rate (in bpm).

    Returns:
    - hr_bounds (tuple): Lower and upper acceptable bounds for HR (mean ± 3*std).
    - pr_bounds (tuple): Lower and upper acceptable bounds for PR (mean ± 3*std).
    """

    # Limits for outliers detection
    hr_mean = np.mean(hr_final)
    hr_std = np.std(hr_final)

    pr_mean = np.mean(pr_final)
    pr_std = np.std(pr_final)
    
    n_std = 3
    hr_upper_bound = hr_mean + n_std * hr_std
    hr_lower_bound = hr_mean - n_std * hr_std

    pr_upper_bound = pr_mean + n_std * pr_std
    pr_lower_bound = pr_mean - n_std * pr_std

    hr_bounds = (hr_lower_bound, hr_upper_bound)
    pr_bounds = (pr_lower_bound, pr_upper_bound)

    return hr_bounds, pr_bounds


# =============================================================================
# WINDOW REMOVAL FUNCTION
# =============================================================================

def extract_good_windows(
    hr_final, time_hr, pr_final, time_pr,
    diff_hr_pr, time_common, diff_peaks,
    hr_bounds, pr_bounds, window_sec, th_n_outliers, th_n_peaks
):
    """
    Extract and reconstruct 'good' HR and PR signal segments,
    based on outlier thresholds and peak difference filtering.

    Parameters:
    - hr_final (array): ECG-derived Heart Rate (in bpm).
    - time_hr (array): Time vector for HR.
    - pr_final (array): PPG-derived Pulse Rate (in bpm).
    - time_pr (array): Time vector for PR.
    - diff_hr_pr (array): Absolute difference |HR_ecg - PR_ppg|.
    - time_common (array): Time vector for diff_hr_pr.
    - diff_peaks (array): Indices of peaks in diff_hr_pr.
    - hr_bounds (tuple): Acceptable (min, max) bounds for HR.
    - pr_bounds (tuple): Acceptable (min, max) bounds for PR.
    - window_sec (int): Size of the sliding window in seconds.
    - th_n_outliers (int): Maximum number of outliers allowed in a window to be considered clean.
    - th_n_peaks (int): Maximum number of diff_hr_pr peaks allowed in a bad window.

    Returns:
    - hr_clean_total (array): Clean HR signal concatenated across good windows.
    - pr_clean_total (array): Clean PR signal concatenated across good windows.
    - time_hr_clean_no_holes (array): Reconstructed HR time vector (no gaps).
    - time_pr_clean_no_holes (array): Reconstructed PR time vector (no gaps).
    - windows_ranges (list of tuples): Start and end times of each window.
    - good_windows (list of bool): Flags for good/bad window status.
    - n_peaks_per_window (list of int): Number of diff_hr_pr peaks per window.
    """
    
    good_windows = []
    windows_ranges = []
    n_peaks_per_window = []
    start_time = 0

    # 1. Identification of good signal windows
    hr_lower_bound, hr_upper_bound = hr_bounds
    pr_lower_bound, pr_upper_bound = pr_bounds

    # Loop through the signal using a sliding window approach
    while start_time + window_sec <= min(time_hr[-1], time_pr[-1]):
        end_time = start_time + window_sec

        # Extract HR and PR data within the current time window
        mask_hr = (time_hr >= start_time) & (time_hr < end_time)
        window_hr = hr_final[mask_hr]

        mask_pr = (time_pr >= start_time) & (time_pr < end_time)
        window_pr = pr_final[mask_pr]

        # Count how many diff_hr_pr peaks fall within this window (allowing ±0.5s margin)
        peaks = (time_common[diff_peaks] >= start_time - 0.5) & (time_common[diff_peaks] <= end_time + 0.5)
        n_peaks_per_window.append(np.sum(peaks))
        
        # Skip windows that have no data in either HR or PR
        if len(window_hr) == 0 or len(window_pr) == 0:
            start_time = end_time
            continue

        # Count outliers in HR and PR signal for the current window
        count_out_hr = np.sum((window_hr < hr_lower_bound) | (window_hr > hr_upper_bound))
        count_out_pr = np.sum((window_pr < pr_lower_bound) | (window_pr > pr_upper_bound))

        # Define the "good window" condition:
        #  - Case 1: Window has few outliers in both signals → always good
        #  - Case 2: If there are outliers, the window is still good if it contains few diff_hr_pr peaks
        # This allows retaining windows with matching abnormal points (i.e., same peaks in both signals)
        if (count_out_hr < th_n_outliers) and (count_out_pr < th_n_outliers):
            is_good = True
        else:
            is_good = peaks.sum() < th_n_peaks

        good_windows.append(is_good)
        windows_ranges.append((start_time, end_time))
        
        start_time += window_sec
        
    # Ensure all elements are bools
    good_windows = [bool(val) for val in good_windows]
    
    n_good = sum(good_windows)
    n_total = len(good_windows)
    print(f"Good windows: {n_good}/{n_total} ({(n_good/n_total)*100:.2f}%)")


    # 2. Concatenation of HR/PR signal good zones and associated reconstructed time
    hr_clean, pr_clean = [], []
    time_hr_clean, time_pr_clean = [], []
    
    # Initialize time_shift to keep track of the cumulative time lost due to removal of bad windows
    time_shift = 0
    
    # Iterate over each window and its quality label (is_good = True/False)
    for (start, end), is_good in zip(windows_ranges, good_windows):
        if is_good:
            mask_hr = (time_hr >= start) & (time_hr < end)
            mask_pr = (time_pr >= start) & (time_pr < end)
    
            # Proceed only if both HR and PR have data in this window
            if np.any(mask_hr) and np.any(mask_pr):
                # Append signal segments to the clean lists
                hr_clean.append(hr_final[mask_hr])
                pr_clean.append(pr_final[mask_pr])
                
                # Time vectors
                time_hr_seg = time_hr[mask_hr]
                time_pr_seg = time_pr[mask_pr]
    
                # Shift the time vectors to remove gaps caused by discarded (bad) windows
                time_hr_clean.append(time_hr_seg - time_shift)
                time_pr_clean.append(time_pr_seg - time_shift)
        else:
            # If the window is bad, increase the time shift by the window duration (gap will be removed)
            time_shift += (end - start)  
    
    hr_clean_total = np.concatenate(hr_clean)
    pr_clean_total = np.concatenate(pr_clean)
    time_hr_clean_no_holes = np.concatenate(time_hr_clean)
    time_pr_clean_no_holes = np.concatenate(time_pr_clean)
    
    return hr_clean_total, pr_clean_total, time_hr_clean_no_holes, time_pr_clean_no_holes, windows_ranges, good_windows, n_peaks_per_window
 

# =============================================================================
# PLOT FUNCTION
# =============================================================================   

def plot_hr_pr_before_after_removing(
    time_hr_original, ecg_hr_original,
    time_pr_original, ppg_hr_original,
    time_hr_clean, hr_clean,
    time_pr_clean, pr_clean,
    subject_id, window_sec, th_n_outliers, th_n_peaks
):
    """
    Plot original and cleaned heart rate/pulse rate (after windows removing) signals from ECG and PPG,
    and show percentage and time duration of removed data.

    Parameters:
    - time_hr_original (array): Time vector for original HR.
    - ecg_hr_original (array): Original HR from ECG.
    - time_pr_original (array): Time vector for original PR.
    - ppg_hr_original (array): Original PR from PPG.
    - time_hr_clean (array): Time vector for cleaned HR.
    - hr_clean (array): Cleaned HR from ECG.
    - time_pr_clean (array): Time vector for cleaned PR.
    - pr_clean (array): Cleaned PR from PPG.
    - subject_id (str): Identifier for the subject (for title).
    - window_sec (int): Duration of each window in seconds (for title).
    - th_n_outliers (int): Maximum number of outliers (for title).
    - th_n_peaks (int): Maximum number of diff_hr_pr peaks (for title).
    """

    # Calculate total durations (in hours and minutes) 
    sec_hr_original = min(time_hr_original[-1] - time_hr_original[0], 
                          time_pr_original[-1] - time_pr_original[0])
    sec_hr_clean = min(time_hr_clean[-1] - time_hr_clean[0], 
                       time_pr_clean[-1] - time_pr_clean[0])
    removed_percent = (sec_hr_original - sec_hr_clean) / sec_hr_original * 100

    hours_original, rem = divmod(sec_hr_original, 3600)
    minutes_original, _ = divmod(rem, 60)

    hours_clean, rem = divmod(sec_hr_clean, 3600)
    minutes_clean, _ = divmod(rem, 60)

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    fig.suptitle(f'HR: {removed_percent:.2f}% removed ({window_sec}sec, th_outliers={th_n_outliers}, th_peaks_diffHR={th_n_peaks}) - {subject_id}', fontsize=16, fontweight="bold")

    # Original HR/PR plot
    axes[0].plot(time_pr_original, ppg_hr_original, color='blue', label='PR (from PPG)')
    axes[0].plot(time_hr_original, ecg_hr_original, color='red', label='HR (from ECG)')
    axes[0].set_ylabel("bpm")
    axes[0].set_title(f"Original Heart/Pulse Rate - {int(hours_original)}h {int(minutes_original)}m")
    axes[0].legend(loc='upper right')

    # Cleaned HR/PR plot
    axes[1].plot(time_pr_clean, pr_clean, color='blue', label='PR (from PPG)')
    axes[1].plot(time_hr_clean, hr_clean, color='red', label='HR (from ECG)')
    axes[1].set_ylabel("bpm")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title(f"Cleaned Heart/Pulse Rate - {int(hours_clean)}h {int(minutes_clean)}m")
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()



def plot_hr_pr_and_bad_windows(
    time_hr, hr_final, hr_bounds,
    time_pr, pr_final, pr_bounds, 
    good_windows, windows_ranges,
    time_ecg, sleep_ecg, ecg_peaks,
    time_ppg, sleep_ppg, ppg_peaks,
    diff_hr_pr, time_common, diff_peaks,
    n_peaks_per_window,
    subject_id, window_sec, th_n_outliers, th_n_peaks
):
    """
    Plot ECG and PPG signals, HR and PR values, diff_hr_pr with Peaks and highlight 
    considered bad windows.
    
    Parameters:
    - time_hr (array): Time vector for HR.
    - hr_final (array): ECG-derived Heart Rate (in bpm).
    - hr_bounds (tuple): Acceptable (min, max) bounds for HR.
    - time_pr (array): Time vector for PPG HR.
    - pr_final (array): PPG-derived Pulse Rate (in bpm).
    - pr_bounds (tuple): Acceptable (min, max) bounds for PR.
    - good_windows (list of bool): Flags for good/bad window status.
    - windows_ranges (list of tuples): Start and end times of each window.
    - time_ecg (array): Time vector for ECG.
    - sleep_ecg (array): ECG signal.
    - ecg_peaks (array): Peaks positions for ECG.
    - time_ppg (array): Time vector for PPG.
    - sleep_ppg (array): PPG signal.
    - ppg_peaks (array): Peaks positions for PPG.
    - diff_hr_pr (array): Absolute difference |HR_ecg - PR_ppg|.
    - time_common (array): Time vector for diff_hr_pr.
    - diff_peaks (array): Indices of peaks in diff_hr_pr.      
    - subject_id (str): Identifier for the subject (for title).
    - window_sec (int): Duration of each window in seconds (for title).
    - th_n_outliers (int): Maximum number of outliers (for title).
    - th_n_peaks (int): Maximum number of diff_hr_pr peaks (for title).
    """

    # I repeat the number of spikes per window for the duration of each window (only to plot)
    time_peaks_extended = []
    peaks_value_extended = []
    
    start_time = 0
    
    for n_peaks in n_peaks_per_window:
        times = np.arange(start_time, start_time + window_sec, 0.5)  
        values = np.full_like(times, fill_value=n_peaks, dtype=float)  
        
        time_peaks_extended.extend(times)
        peaks_value_extended.extend(values)
        
        start_time += window_sec

    hr_lower_bound, hr_upper_bound = hr_bounds
    pr_lower_bound, pr_upper_bound = pr_bounds
    
    fig, axes = plt.subplots(5, 1, figsize=(20, 15), sharex=True)
    fig.suptitle(f'Signals with Bad Windows ({window_sec}sec, th_outliers={th_n_outliers}, th_peaks_diffHR={th_n_peaks}) - {subject_id}', fontsize=16, fontweight="bold")
    
    # Heart/Pulse Rate
    axes[0].plot(time_pr, pr_final, color='blue', label='PR (from PPG)')
    axes[0].plot(time_hr, hr_final, color='red', label='HR (from ECG)')
    axes[0].axhline(hr_upper_bound, color='red', linestyle='--', label='HR Mean + 3std')
    axes[0].axhline(hr_lower_bound, color='red', linestyle='--', label='HR Mean - 3std')
    axes[0].axhline(pr_upper_bound, color='blue', linestyle='--', label='PR Mean + 3std')
    axes[0].axhline(pr_lower_bound, color='blue', linestyle='--', label='PR Mean - 3std')
    axes[0].set_ylabel("bpm")

    for (start, end), is_good in zip(windows_ranges, good_windows):
        if not is_good:
            axes[0].axvspan(start, end, color='orange', alpha=0.6)
           
    # ECG with Peaks
    axes[1].plot(time_ecg, sleep_ecg, color="orange", label="ECG")
    axes[1].scatter(time_ecg[ecg_peaks], sleep_ecg[ecg_peaks],
                   color="red", marker="o", label="ECG Peaks", zorder=3)
    axes[1].set_ylabel("ECG")
    axes[1].legend(loc='upper right')
    
    # PPG with Peaks
    axes[2].plot(time_ppg, sleep_ppg, color="orange", label="PPG")
    axes[2].scatter(time_ppg[ppg_peaks], sleep_ppg[ppg_peaks],
                  color="red", marker="o", label="PPG Peaks", zorder=3)
    axes[2].set_ylabel("PPG")
    axes[2].legend(loc='upper right')

    # Diff HR/PR and Peaks
    axes[3].plot(time_common, diff_hr_pr, color='black', label='|HR - PR|')
    axes[3].scatter(time_common[diff_peaks], diff_hr_pr[diff_peaks], color='red', label='Peaks in |HR - PR|')
    axes[3].set_ylabel("|HR - PR| (bpm)")
    axes[3].legend(loc='upper right')

    # Number of Peaks diff_hr_pr per Windows 
    axes[4].plot(time_peaks_extended, peaks_value_extended, color='green', label='Number of Peaks in |HR - PR| per Window', drawstyle='steps-post')
    axes[4].set_ylabel("Num Peaks in |HR - PR| per Window")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(loc='upper right')

    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='PR (from PPG)'),
        Line2D([0], [0], color='red', lw=2, label='HR (from ECG)'),
        Line2D([0], [0], color='black', linestyle='--', lw=2, label='Mean ± 3std'),
        Patch(facecolor='orange', edgecolor='orange', alpha=0.5, label='Bad windows'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
   