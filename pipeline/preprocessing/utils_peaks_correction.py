"""
This code provides functions to correct peaks position in ECG and PPG signals. 

After neurokit functions in the main to identify peaks (signal_peaks, signal_fixpeaks), it is 
necessary to further correct the peaks. In particular, the following steps are performed:
- checking that each peak is a local max
- removal of ECG peaks that are too low
- checking that distance between peaks is > d_max
- correction of any negative PPG peaks
- checking that distance between peaks is < d_min
"""


import numpy as np
from scipy.signal import find_peaks
import neurokit2 as nk
import matplotlib.pyplot as plt


# =============================================================================
# PEAKS CORRECTION FUNCTIONS
# =============================================================================

def peaks_correction(ecg_processed, ppg_processed, ecg_peaks, ppg_peaks, fs_ecg, fs_ppg):
    """
    Function to perform correction of ECG and PPG peaks.
    This function refines the initial peak detection results by applying some correction steps 
    to remove artifacts, recover missing peaks and ensure physiological plausibility. 
 
    Parameters:
    - ecg_processed (array): Processed ECG signal.
    - ppg_processed (array): Processed PPG signal.
    - ecg_peaks (array): Detected ECG peak indices.
    - ppg_peaks (array): Detected PPG peak indices.
    - fs_ecg (float): ECG sampling frequency.
    - fs_ppg (float): PPG sampling frequency.

    Returns:
    - ecg_peaks_final (array): Corrected ECG peak indices.
    - ppg_peaks_final (array): Corrected PPG peak indices.
    """

    # 1) Correct Local Max: ensure that each detected peak is a local maximum within a defined window
    ecg_peaks_loc_max = correct_local_maxima(ecg_processed, ecg_peaks, window_size=120)
    ppg_peaks_loc_max = correct_local_maxima(ppg_processed, ppg_peaks, window_size=120)

    # 2) Elimination of excessively low ECG peak values 
    ecg_peaks_clean = remove_weak_ecg_peaks(ecg_processed, ecg_peaks_loc_max, fs_ecg)

    # Calculate initial HR from ECG and PR from PPG
    hr = nk.ecg_rate(ecg_peaks_clean, sampling_rate=fs_ecg, desired_length=len(ecg_processed))
    pr = nk.ppg_rate(ppg_peaks_loc_max, sampling_rate=fs_ppg, desired_length=len(ppg_processed))

    # 3) Correct Max Distance: find missing peaks if the distance between consecutive peaks > max_dist
    max_dist = 1.7  # in seconds (≈35 bpm)
    ecg_peaks_dmax = correct_dmax(ecg_processed, hr, ecg_peaks_clean, fs_ecg, max_dist)
    ppg_peaks_dmax = correct_dmax(ppg_processed, pr, ppg_peaks_loc_max, fs_ppg, max_dist)

    # 4) Corrects negative PPG peaks by searching for alternative peaks between neighboring peaks
    ppg_peaks_clean = correct_negative_ppg_peaks(ppg_processed, ppg_peaks_dmax)

    # 5) Correct Min Distance: remove peaks that are too close to each other while keeping the highest value
    bpm = 120  
    min_distance_sec = 60 / bpm  # Convert BPM to minimum allowed peak distance (in seconds)
    min_dist_ecg = int(min_distance_sec * fs_ecg)  
    min_dist_ppg = int(min_distance_sec * fs_ppg)  
    ecg_peaks_final = correct_dmin(ecg_processed, ecg_peaks_dmax, min_dist_ecg)
    ppg_peaks_final = correct_dmin(ppg_processed, ppg_peaks_clean, min_dist_ppg)

    return ecg_peaks_final, ppg_peaks_final


# =============================================================================
# Helper Functions (Peaks Correction)
# =============================================================================

def correct_local_maxima(signal, peaks, window_size):
    """
    Corrects peaks that are not local maxima by replacing them with the highest value 
    within a surrounding window.
    
    Parameters:
    - signal (array): The signal in which peaks are detected.
    - peaks (array): Detected peak indices.
    - param window_size: The size of the window used to search for the local maximum.
    
    Returns:
    - corrected_peaks (array): Corrected peak indices.
    """
    
    corrected_peaks = []

    # Add the first peak (it has no previous peak to compare)
    corrected_peaks.append(peaks[0])

    # Iterate through peaks
    for i in range(1, len(peaks) - 1):
        curr_peak = peaks[i]

        # Define the search window around the current peak
        start = max(curr_peak - window_size, 0)
        end = min(curr_peak + window_size + 1, len(signal))

        # Find the true peaks within the window
        local_peaks, _ = find_peaks(signal[start:end])

        # If there are peaks, select the highest one
        if len(local_peaks) > 0:  
            max_peak_relative = local_peaks[np.argmax(signal[start:end][local_peaks])]
            max_peak_index = start + max_peak_relative  # Map the index to the global signal
            corrected_peaks.append(max_peak_index)
        else:
            # If no valid peaks are found in the window, do not add anything
            continue

    # Add the last peak (it has no next peak to compare)
    corrected_peaks.append(peaks[-1])

    return np.array(corrected_peaks)



def remove_weak_ecg_peaks(ecg, ecg_peaks, fs_ecg, window_size_sec=30, threshold_ratio=0.3):
    """
    Removes excessively low ECG peak values caused by noise, using a sliding window approach.

    Parameters:
    - ecg (array): ECG signal.
    - ecg_peaks (array): Detected ECG peak indices.
    - fs_ecg (int): ECG sampling frequency.
    - window_size_sec (int, optional): Window size in seconds for local thresholding (default is 30 seconds).
    - threshold_ratio (float, optional): Ratio to determine local threshold for weak peak removal (default is 0.3).

    Returns:
    - corrected_peaks (array): Corrected peaks indices.
    """
    
    window_samples = int(window_size_sec * fs_ecg)  
    corrected_peaks = []

    # Slide through the ECG signal with the given window
    for start in range(0, len(ecg), window_samples):
        end = start + window_samples

        # Select peaks within the current window
        peaks_in_window = ecg_peaks[(ecg_peaks >= start) & (ecg_peaks < end)]

        if len(peaks_in_window) == 0:
            continue  # Skip if no peaks are found in this window

        # Compute local threshold based on mean amplitude of peaks in the window
        local_threshold = np.mean(ecg[peaks_in_window]) * threshold_ratio

        # Identify peaks below the local threshold (likely noise) and keep only the valid peaks
        peaks_to_remove = peaks_in_window[ecg[peaks_in_window] < local_threshold]
        peaks_filtered = np.setdiff1d(peaks_in_window, peaks_to_remove)

        # Add the filtered peaks to the cleaned list
        corrected_peaks.extend(peaks_filtered)

    return np.array(corrected_peaks)



def correct_dmax(signal, heart_rate, peaks, fs, max_dist):
    """
    Corrects missing peaks by adding new ones if the distance between consecutive peaks 
    exceeds a given threshold (max_dist).

    Parameters:
    - signal (array): The signal in which peaks are detected.
    - heart_rate (array): The heart rate found with the previous peak detection.
    - peaks (array): Detected peaks indices.
    - fs (int or float): Sampling frequency.
    - max_dist (float): Maximum allowed distance (in seconds) between consecutive peaks.

    Returns:
    - corrected_peaks (array): Corrected peaks indices.
    """
    
    new_peaks = []

    # Iterate through the peaks, excluding the last one, to check distances between consecutive peaks
    for i in range(len(peaks) - 1):
        # Calculate the distance between consecutive peaks in seconds
        distance = (peaks[i + 1] - peaks[i]) / fs  

        # If the distance exceeds max_dist, we try to find intermediate peaks
        if distance > max_dist:
            start, end = peaks[i], peaks[i + 1]  
            segment = signal[start:end]

            # Calculate the distance and the height for intermediate peaks based on the average heart rate
            dist = (60 / np.mean(heart_rate)) * 0.7
            h = (0.1 * np.mean(np.array([signal[peaks[i]], signal[peaks[i+1]]])))

            # Find intermediate peaks within the segment with a minimum distance and height between them
            intermediate_peaks, _ = find_peaks(segment, distance=int(dist * fs), height=h)

            # Adjust the indices to match the original signal by adding the start index of the segment
            intermediate_peaks_corrected = intermediate_peaks + start

            new_peaks.extend(intermediate_peaks_corrected)

    # Combine the original peaks with the new intermediate peaks and sort them
    corrected_peaks = np.sort(np.concatenate((peaks, new_peaks)))

    return np.array(corrected_peaks, dtype=int)



def correct_negative_ppg_peaks(ppg, ppg_peaks):
    """
    Corrects negative PPG peaks by searching for alternative peaks between neighboring peaks.
    
    For each negative peak, the function looks for valid replacement peaks both:
    - Between the previous peak and the current negative peak
    - Between the current negative peak and the next peak
    
    If valid peaks are found, they are added to the final list and the original negative peak is removed.

    Parameters:
    - ppg (array): PPG signal.
    - ppg_peaks (array): Detected PPG peak indices.
    
    Returns:
    - corrected_peaks (array): Corrected peaks indices.
    """

    # Find indices of negative peaks
    negative_peak_indices = np.where(ppg[ppg_peaks] < 0)[0]

    corrected_peaks = ppg_peaks.copy()
    values_to_remove = []

    for i in negative_peak_indices:
        peak_idx = ppg_peaks[i]

        # Ensure we are not at the signal boundaries
        if i > 0 and i < len(corrected_peaks) - 1:
            prev_peak = corrected_peaks[i - 1]
            next_peak = corrected_peaks[i + 1]

            # Minimum height based on adjacent peaks
            min_height = 0.05 * np.mean([ppg[prev_peak], ppg[next_peak]])

            new_candidates = []

            # Search for peaks between previous peak and current negative peak
            intermediate_peaks1, _ = find_peaks(ppg[prev_peak:peak_idx], height=min_height)
            if len(intermediate_peaks1) > 0:
                # Adjust indices to match the original signal
                new_candidates.extend(intermediate_peaks1 + prev_peak)

            # Search for peaks between current negative peak and next peak
            intermediate_peaks2, _ = find_peaks(ppg[peak_idx:next_peak], height=min_height)
            if len(intermediate_peaks2) > 0:
                # Adjust indices to match the original signal
                new_candidates.extend(intermediate_peaks2 + peak_idx)

            if new_candidates:
                corrected_peaks = np.concatenate((corrected_peaks, new_candidates))

            # Mark the original negative peak for removal
            values_to_remove.append(peak_idx)

        else:
            # If at the boundary, just mark the negative peak for removal
            values_to_remove.append(peak_idx)

    # Remove the original negative peaks
    corrected_peaks = corrected_peaks[~np.isin(corrected_peaks, values_to_remove)]
    corrected_peaks = np.sort(corrected_peaks)

    return corrected_peaks
    
    
    
def correct_dmin(signal, peaks, min_dist):
    """
    Removes peaks that are too close together by keeping only the highest one.

    Parameters:
    - signal (array): The signal in which peaks are detected.
    - peaks (array): Detected peaks indices.
    - min_distance (int): Minimum allowed distance between peaks (in samples).

    Returns:
    - corrected_peaks (array): Corrected peaks indices.
    """
    
    if len(peaks) < 2:
        return peaks  
    
    peaks = np.array(peaks, dtype=int)
    
    corrected_peaks = [peaks[0]]  # Start with the first peak
    
    for i in range(1, len(peaks)):
        if peaks[i] - corrected_peaks[-1] < min_dist:
            # If peaks are too close, keep the one with the highest amplitude
            if signal[peaks[i]] > signal[corrected_peaks[-1]]:
                corrected_peaks[-1] = peaks[i]  # Replace with the higher peak
        else:
            corrected_peaks.append(peaks[i])  # Keep the peak if it's far enough
    
    return np.array(corrected_peaks)



# =============================================================================
# PLOT FUNCTION
# =============================================================================

def plot_signals_peaks(time_ecg, ecg, ecg_peaks, time_ppg, ppg, ppg_peaks, hr, pr, subj):
    """
    Function to plot ECG and PPG signals with detected peaks and corresponding heart/pulse rates.

    Parameters:
    - time_ecg (array): Time vector corresponding to the ECG signal.
    - ecg (array): ECG signal.
    - ecg_peaks (array): Detected ECG peak indices.
    - time_ppg (array): Time vector corresponding to the PPG signal.
    - ppg (array): PPG signal.
    - ppg_peaks (array): Detected PPG peak indices.
    - hr (array): Heart Rate (bpm) computed from ECG peaks.
    - pr (array): Pulse Rate (bpm) computed from PPG peaks.
    - subj (str): Subject identifier (used for plot title).

    Returns:
    - None: Displays a figure with three subplots:
        1. ECG signal with R-peaks.
        2. PPG signal with systolic peaks.
        3. Heart Rate and Pulse Rate trends over time.
    """

    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
    fig.suptitle(f'Final Detection - {subj}', fontsize=16, fontweight="bold")
    
    axes[0].plot(time_ecg, ecg, label="ECG", color="orange")
    axes[0].scatter(time_ecg[ecg_peaks], ecg[ecg_peaks], color="red", label="R-Peaks", zorder=3)
    axes[0].set_ylabel("ECG")
    axes[0].legend(loc='upper right')
    
    axes[1].plot(time_ppg, ppg, label="PPG", color="orange")
    axes[1].scatter(time_ppg[ppg_peaks], ppg[ppg_peaks], color="red", label="Systolic Peaks", zorder=3)
    axes[1].set_ylabel("PPG")
    axes[1].legend(loc='upper right')
    
    axes[2].plot(time_ecg, hr, label='Heart Rate (ECG)', color='red')
    axes[2].plot(time_ppg, pr, label='Pulse Rate (PPG)', color='blue')
    axes[2].set_ylabel("bpm")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Heart Rate vs Pulse Rate Over Time")
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()