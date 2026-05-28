"""
This code provides functions to load, pre-process, clean ECG and PPG signals,
including unit conversion, resampling, ECG inversion correction, 
filtering, flat zone removal and handling of hypnogram-based sleep intervals.
"""


import numpy as np
import neurokit2 as nk
import math
import datetime 
from scipy.signal import resample
from pyedflib import highlevel


# =============================================================================
# LOAD SIGNALS FUNCTIONS
# =============================================================================

def load_signals(path_edf, path_hyp, subj, subj_nox):
    """
    Function to load ECG and PPG signals from EDF and corresponding hypnogram from TXT files.
    This function extracts ECG and PPG signals, their sampling frequencies and units from edf
    and load the hypnogram. 

    Parameters:
    - path_edf (str): Path to the .edf file containing physiological signals.
    - path_hyp (str): Path to the .txt file containing the sleep hypnogram.
    - subj (str): Subject identifier.
    - subj_nox (list): List of subjects with Nox device.

    Returns:
    - ecg (array): ECG signal.
    - ppg (array): PPG signal.
    - fs_ecg (float): ECG sampling frequency.
    - fs_ppg (float): PPG sampling frequency.
    - unit_ecg (str): ECG unit of measurement.
    - unit_ppg (str): PPG unit of measurement.
    - hyp_expanded (array): Hypnogram expanded to sample-level.
    """

    # Load File.edf 
    signals_edf, signal_headers, _ = highlevel.read_edf(path_edf)

    ecg_index = next((i for i, h in enumerate(signal_headers) if "ECG" in h['label']), None)
    ppg_index = next((i for i, h in enumerate(signal_headers) if "Pleti" in h['label'] or "Pulse Waveform" in h['label']), None)

    if ecg_index is None or ppg_index is None:
        raise ValueError(f"ECG or PPG signal not found in {path_edf}")
    
    ecg = signals_edf[ecg_index]  
    ppg = signals_edf[ppg_index]

    fs_ecg = signal_headers[ecg_index]['sample_frequency']
    fs_ppg = signal_headers[ppg_index]['sample_frequency']

    unit_ecg = signal_headers[ecg_index]['dimension']
    unit_ppg = signal_headers[ppg_index]['dimension']


    # Load File.txt 
    totDuration = len(ecg) / fs_ecg 

    if subj in subj_nox:
        hyp = readscoreTurin_Nox(path_hyp, totDuration)
    else:
        hyp = readscoreTurin_Somtè(path_hyp, totDuration)

    # Expand hypnogram from epochs (30s) to sample-level
    epoch_length = 30 * fs_ecg
    hyp_expanded = np.repeat(hyp, epoch_length)

    return ecg, ppg, fs_ecg, fs_ppg, unit_ecg, unit_ppg, hyp_expanded


# =============================================================================
# Helper Functions (Load Signals)
# =============================================================================

"""
    The function reads the txt file that contains polysomnography annotations 
    from the database and returns the row vector hyp. Each element of the 
    array represents a 30-second epoch, scored following AASM rules.
    Different approch for Somtè and Nox device (different txt file structure).

    Score mapping:
    '?' -> Not scored epochs (before sleep-time or after sleep ends) -> NaN (not present in Nox)
    'W/Veglia' -> 0 (Wake)
    'N1' -> 1 (Light Sleep N1)
    'N2' -> 2 (Light Sleep N2)
    'N3' -> 3 (Deep Sleep N3)
    'R/REM' -> 5 (REM Sleep)

    Parameters:
    - path (str): Path to the annotation file (.txt).
    - totDuration (float): Total record duration in seconds.

    Returns:
    - hyp (numpy array): Score array, each element is a 30s sleep epoch.
"""

def readscoreTurin_Somtè(path, totDuration):
    
    hyp = []
    # Read annotation file
    with open(path, 'r') as f:
        for line in f:
            tline = line.strip()
            if tline == '?':
                hyp.append(np.nan)
            elif tline == 'W':
                hyp.append(0)
            elif tline == 'N1':
                hyp.append(1)
            elif tline == 'N2':
                hyp.append(2)
            elif tline == 'N3':
                hyp.append(3)
            elif tline == 'R':
                hyp.append(5)

    hyp = np.array(hyp)

    return hyp 


    
def readscoreTurin_Nox(path, totDuration):

    start_epoch = []
    sleep_phase = []

    # Open the file in read mode
    with open(path, 'r') as file:
        # Skip the first two lines
        file.readline()
        file.readline()

        # Read the rest of the file
        for line in file:
            columns = line.strip().split('\t')  # Split the line into columns
            start_epoch.append(int(columns[1]))
            sleep_phase.append(columns[2])

    # It is necessary to insert NaN at the first epochs (not scored)
    num = int(start_epoch[0]) - 1  # Number epochs before those scored
    hyp = [np.nan] * num
    
    # Assign values to 'hyp' based on sleep phase and check epoch continuity
    previous_epoch = start_epoch[0]  # Set the first epoch as the starting reference
    
    for i, phase in enumerate(sleep_phase):
        # Check if the current start_epoch is not consecutive
        while previous_epoch < start_epoch[i] - 1:
            hyp.append(np.nan)  # Add NaN for missing epochs
            previous_epoch += 1  # Move to the next expected epoch
    
        if phase == 'Veglia':
            hyp.append(0)
        elif phase == 'N1':
            hyp.append(1)
        elif phase == 'N2':
            hyp.append(2)
        elif phase == 'N3':
            hyp.append(3)
        elif phase == 'REM':
            hyp.append(5)
    
        # Update previous_epoch
        previous_epoch = start_epoch[i]
    
    # Calculate how many NaN values to add at the end to reach the desired length
    num_extra_nans = math.ceil(totDuration / 30 - len(hyp))
    hyp.extend([np.nan] * num_extra_nans)

    return hyp


# =============================================================================
# PRE-PROCESSING FUNCTIONS
# =============================================================================
 
def preprocessing(ecg, ppg, fs_ecg, fs_ppg, unit_ecg, unit_ppg, hyp, subj, subj_nox):
    """
    Function to perform pre-processing of ECG and PPG signals before peak detection,
    including unit normalization, resampling, signal inversion correction, filtering 
    and near-zero amplitude segments removal.

    Parameters:
    - ecg (array): Raw ECG signal.
    - ppg (array): Raw PPG signal.
    - fs_ecg (float): ECG sampling frequency.
    - fs_ppg (float): PPG sampling frequency.
    - unit_ecg (str): ECG unit of measurement.
    - unit_ppg (str): PPG unit of measurement.
    - hyp (array): Hypnogram data used to select valid sleep intervals.
    - subj (str): Subject identifier.
    - subj_nox (list): List of subjects with Nox device.

    Returns:
    - sleep_ecg_clean (array): Pre-processed ECG signal.
    - sleep_ppg_clean (array): Pre-processed PPG signal.
    """

    # Signal Pre-processing 
    # Unit of measurement: conversion to mV 
    ecg, ppg = unit_of_measurement(subj, subj_nox, ecg, ppg, unit_ecg, unit_ppg)   

    # Resample (for Nox signals: fs_ecg is different from fs_ppg)
    ecg, ppg, fs_ecg, fs_ppg = resample_if_necessary(ecg, ppg, fs_ecg, fs_ppg)      

    # Verify if ECG is inverted
    ecg = ecg_is_inverted(ecg, fs_ecg)


    # Removes the intervals where the hypnogram is NaN or 0 
    sleep_ecg, sleep_ppg, sleep_hyp = remove_intervals(fs_ecg, hyp, ecg, ppg)


    # Signal Cleaning 
    # ECG: bandpass 5-30 Hz Butterworth 
    # PPG: bandpass 0.5-8 Hz Butterworth (Elgendi method neurokit)
    ecg_filtered = nk.signal_filter(sleep_ecg, fs_ecg, lowcut=5, highcut=30, method='butterworth', order=4) 
    ppg_filtered = nk.ppg_clean(sleep_ppg, fs_ppg, method='elgendi')
        

    # Detect and remove flat zones from ECG and PPG signals 
    sleep_ecg_clean, sleep_ppg_clean = remove_flat_zones(ecg_filtered, ppg_filtered, fs_ecg, fs_ppg, min_sec=1, expansion_sec=5)

    return sleep_ecg_clean, sleep_ppg_clean


# =============================================================================
# Helper Functions (PreProcessing)
# =============================================================================

def unit_of_measurement(subj, subj_nox, ecg, ppg, unit_ecg, unit_ppg):
    """
    Function to handle the unit conversion for ECG and PPG (useful for Nox data).
    
    Parameters:
    - subj (str): Subject identifier.
    - subj_nox (list): List of subjects with Nox device.
    - ecg (array): ECG signal.
    - ppg (array): PPG signal.
    - unit_ecg (str): ECG unit of measurement ('V' expected for ECG in Nox).
    - unit_ppg (str): PPG unit of measurement ('' expected for PPG in Nox).
    
    Returns:
    - ecg (array): Modified ECG signal if necessary.
    - ppg (array): Modified PPG signal if necessary.
    """
    
    if subj in subj_nox:
        print('Nox device')
        if unit_ecg == 'V':
            ecg = ecg * 1000
        else:
            print(f"Unexpected ECG unit '{unit_ecg}'. Values may be incorrect.")
            
        # a.u.
        if unit_ppg == '':
            ppg = ppg / 10000
        else:
            print(f"Unexpected PPG unit '{unit_ppg}'. Values may be incorrect.")
                   
    return ecg, ppg
            


def resample_if_necessary(ecg, ppg, fs_ecg, fs_ppg):
    """
    Function to resample ECG and PPG data if the sampling frequencies of ECG and PPG are different.
    
    Parameters:
    - ecg (array): ECG signal.
    - ppg (array): PPG signal.
    - fs_ecg (float): ECG sampling frequency.
    - fs_ppg (float): PPG sampling frequency.
    
    Returns:
    - ecg_resampled (array): Resampled ECG signal.
    - ppg_resampled (array): Resampled PPG signal.
    - fs_ecg (float): Updated ECG sampling frequency.
    - fs_ppg (float): Updated PPG sampling frequency.
    """
    
    if fs_ecg != fs_ppg:
        # Update the new sampling frequency
        f_new = 256  
        
        # Calculate the number of samples needed for the resampled PPG and ECG
        num_samples_ecg = int(len(ecg) * (f_new / fs_ecg))
        num_samples_ppg = int(len(ppg) * (f_new / fs_ppg))
        
        ecg_resampled = resample(ecg, num_samples_ecg)
        ppg_resampled = resample(ppg, num_samples_ppg)
        
        fs_ecg = fs_ppg = f_new
        print('Resampling done!')
        
        return ecg_resampled, ppg_resampled, fs_ecg, fs_ppg
    
    # If no resampling is needed, return the original PPG and sampling frequency
    return ecg, ppg, fs_ecg, fs_ppg



def ecg_is_inverted(ecg, fs_ecg):
    """
    Function to check and invert the ECG signal if it is inverted.
    
    Parameters:
    - ecg (array): ECG signal.
    - fs_ecg (float): ECG sampling frequency.
    
    Returns:
    - ecg (array): ECG signal, inverted if necessary.
    """
    
    # Check if the ECG signal is inverted using the first samples
    _, is_inverted = nk.ecg_invert(ecg[:100000], sampling_rate=fs_ecg)
    
    # If the ECG signal is inverted, invert it
    if is_inverted:
        ecg = - ecg
        print('ECG Inverted!')
    
    return ecg



def remove_intervals(fs_ecg, hyp_expanded, ecg, ppg):
    """
    Removes the intervals where the hypnogram is NaN or 0.

    Parameters:
    - fs_ecg (float): The sampling frequency of the ECG signal.
    - hyp_expanded (array): The expanded hypnogram signal.
    - ecgs (array): ECG signal.
    - ppg (array): PPG signal.

    Returns:
    - sleep_ecg (array): The ECG signal after removing NaN and wake intervals.
    - sleep_ppg (array): The PPG signal after removing NaN and wake intervals.
    - sleep_hyp (array): The hypnogram signal after removing NaN and wake intervals.
    """
    
    # Ensure that the hypnogram length matches the ECG signal length
    if len(hyp_expanded) != len(ecg):
        print(f"{(len(hyp_expanded))/fs_ecg} seconds in hyp and {(len(ecg))/fs_ecg} seconds in ecg")
        
        # Truncate the hypnogram to match the ECG signal length
        hyp_expanded = hyp_expanded[:len(ecg)]  

    # Identify indices where the hypnogram is NaN or 0
    nan_mask = np.isnan(hyp_expanded)  
    zero_mask = (hyp_expanded == 0)    

    # Create a combined mask 
    mask_to_remove = nan_mask | zero_mask

    # Remove the unwanted intervals from the signals by inverting the mask (~) 
    # and keeping only the values that are not NaN or 0 (sleep intervals)
    sleep_ecg = ecg[~mask_to_remove]  
    sleep_ppg = ppg[~mask_to_remove]  
    sleep_hyp = hyp_expanded[~mask_to_remove] 

    return sleep_ecg, sleep_ppg, sleep_hyp



def remove_flat_zones(ecg, ppg, fs_ecg, fs_ppg, min_sec=1, expansion_sec=5):
    """
    Detect and remove flat zones (amplitude remains near zero for at least a given minimum 
    duration) from ECG and PPG signals.

    Parameters:
    - ecg (array): ECG signal.
    - ppg (array): PPG signal.
    - fs_ecg (float): ECG sampling frequency.
    - fs_ppg (float): PPG sampling frequency.
    - min_sec (float, optional): Minimum duration (in seconds) for a flat zone 
      to be considered significant. Default is 1 second.
    - expansion_sec (float, optional): Time (in seconds) to expand around 
      each detected flat zone to ensure full removal. Default is 5 seconds.

    Returns:
    - ecg_clean (array): The ECG signal with flat zones removed.
    - ppg_clean (array): The PPG signal with flat zones removed.
    """
      
    # Detect flat zones in ECG and PPG
    ecg_flat = find_flat_zones(ecg, fs_ecg, min_sec)
    ppg_flat = find_flat_zones(ppg, fs_ppg, min_sec)

    # Expand detected zones by `expansion_sec` seconds on both sides
    ecg_expanded = expand_segments(ecg_flat, fs_ecg, expansion_sec, len(ecg))
    ppg_expanded = expand_segments(ppg_flat, fs_ppg, expansion_sec, len(ppg))

    # Compute total duration of removed regions
    ecg_seconds = count_total_seconds(ecg_expanded, fs_ecg)
    ppg_seconds = count_total_seconds(ppg_expanded, fs_ppg)

    print(f"Removed Zones in ECG: {ecg_seconds:.2f} s ({ecg_seconds/60:.2f} min)")
    print(f"Removed Zones in PPG: {ppg_seconds:.2f} s ({ppg_seconds/60:.2f} min)")

    # Mask and filter
    ecg_mask = create_mask(ecg_expanded, len(ecg))
    ppg_mask = create_mask(ppg_expanded, len(ppg))
    # Combine both masks (OR operation) to create a final mask
    combined_mask = ppg_mask | ecg_mask
    
    # Removes the signal values where the mask is True (flat zones)
    ecg_clean = ecg[~combined_mask]
    ppg_clean = ppg[~combined_mask]

    return ecg_clean, ppg_clean



# --- Helper functions for remove_flat_zones functions ---

def find_flat_zones(signal, fs, min_sec, threshold_min=-1e-5, threshold_max=1e-5): 
    """
    Identify flat zones in a signal where the amplitude stays within a specified range
    around zero for at least a given duration.
    
    Parameters:
    - signal (array): Input signal to be analyzed.
    - fs (int): Sampling frequency.
    - min_sec (float): Minimum duration (in seconds) for a segment to be considered valid.
    - threshold_min (float, optional): Lower threshold for detecting near-zero values. Default is -1e-5.
    - threshold_max (float, optional): Upper threshold for detecting near-zero values. Default is 1e-5.

    Returns:
    - list of tuples: List containing (start, end) indices of the flat segments 
    that meet the duration and threshold criteria.
    """  

    # Create mask: True where the signal value is close to zero
    near_zero = (signal > threshold_min) & (signal < threshold_max)

    # Find changes in the mask (start and end of the segments)
    changes = np.diff(near_zero.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    # Special case: the signal starts with a flat zone
    if near_zero[0]:
        starts = np.insert(starts, 0, 0)
    # Special case: the signal ends with a flat zone
    if near_zero[-1]:
        ends = np.append(ends, len(signal))

    # Filter segments that last at least min_sec seconds
    min_length = fs * min_sec  # Minimum duration in samples
    long_segments = [(start, end) for start, end in zip(starts, ends) if (end - start) >= min_length]

    return long_segments



def expand_segments(segments, fs, expansion_sec, signal_length):
    """
    Expands each interval by adding expansion_sec seconds before and after the segment.
    
    Parameters:
    - segments (list of tuples): List of (start, end) in samples.
    - fs (int): Sampling frequency.
    - expansion_sec (float): Time in seconds to add before and after the segment.
    - signal_length (int): Total length of the signal (to avoid exceeding the boundaries).
    
    Returns:
    - list of tuples: Expanded segments, or an empty list if there are no segments.
    """ 
        
    # If the list is empty
    if not segments:  
        return []
    
    expansion_samples = int(expansion_sec * fs)  
    
    expanded_segments = []
    for start, end in segments:
        new_start = max(0, start - expansion_samples)  # Ensure start does not go below 0
        new_end = min(signal_length, end + expansion_samples)  # Ensure end does not exceed signal length
        expanded_segments.append((new_start, new_end))
    
    return expanded_segments



def count_total_seconds(segments, fs):
    """
    Counts the total number of seconds covered by the segments.
    
    Parameters:
    - segments (list of tuples): List of (start, end) in samples.
    - fs (int): Sampling frequency.
    
    Returns:
    - float: Total number of seconds covered by the segments.
    """

    # Sum the number of samples in all segments
    total_samples = sum(end - start for start, end in segments)  
    # Convert the total samples to seconds
    total_seconds = total_samples / fs  

    return total_seconds



def create_mask(segments, signal_length):
    """
    Creates a mask based on the provided intervals, where True represents 
    the flat zones to be removed, and False represents the valid signal parts.
    
    Parameters:
    - segments (list of tuples): List of intervals (start, end) indicating flat zones.
    - signal_length (int): The length of the signal.
    
    Returns:
    - list: A boolean mask with True for the flat zones, and False for valid signal points.
    """

    # Initialize mask with all False values
    mask = np.zeros(signal_length, dtype=bool)  
    for start, end in segments:
        mask[start:end] = True
    return mask
    


# =============================================================================

def samples_to_hms(signal_length, fs):
    """
    Convert the length of a signal in samples to a duration in hours, minutes and seconds.

    Parameters:
    - signal_length (int): The total number of samples in the signal.
    - fs (float): Sampling frequency.

    Returns:
    - tuple: (hours, minutes, seconds) representing the duration of the signal.
    """
    
    duration_seconds = signal_length / fs
    duration = datetime.timedelta(seconds=duration_seconds)
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    return int(hours), int(minutes), int(seconds)
