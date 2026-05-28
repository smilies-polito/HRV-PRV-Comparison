"""
Code with functions to perform HRV metrics analysis and comparison between the 
various methods.
For plotting and visual representation of results, refer to 'functions_analysis_plot'.
"""


import pandas as pd
from scipy.stats import pearsonr, spearmanr


# Metrics of interest for analysis
# metrics_time = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD"]
# titles_time = ["Mean RR", "SDNN", "RMSSD"]
metrics_time = ["HRV_MeanNN", "HRV_SDNN"]
titles_time = ["Mean NN", "SDNN"]

metrics_freq = ["HRV_LFn", "HRV_HFn", "HRV_LFHF", "HRV_VLF", "HRV_LF", "HRV_HF"]
titles_freq = ["LFn", "HFn", "LF/HF", "VLF", "LF", "HF"]

metrics_nonlinear = ["HRV_ApEn", "HRV_SampEn", "HRV_DFA_alpha1", "HRV_DFA_alpha2", "HRV_SD1", "HRV_SD2", "HRV_SD1SD2"]
titles_nonlinear = ["ApEn", "SampEn", "DFA α₁", "DFA α₂", "SD1", "SD2", "SD1/SD2"]

all_metrics = metrics_time + metrics_freq + metrics_nonlinear
all_titles = titles_time + titles_freq + titles_nonlinear
metric_to_title = dict(zip(all_metrics, all_titles))


# =============================================================================
# COMPUTE CORRELATIONS
# =============================================================================

def compute_correlations_for_metrics(df_hrv, df_prv):
    """
    Calculates Pearson and Spearman correlation coefficients for each metric between HRV and PRV data.

    Parameters:
    - df_hrv (DataFrame): HRV metrics for ECG, with each row corresponding to a window.
    - df_prv (DataFrame): PRV metrics for PPG, with each row corresponding to a window.

    Returns:
    - pearson_results (dict): Dictionary mapping each metric name to its Pearson correlation coefficient.
    - spearman_results (dict): Dictionary mapping each metric name to its Spearman correlation coefficient.
    """
    
    pearson_results = {}
    spearman_results = {}

    for metric in all_metrics:
        # Extract the values for the metric from both the HRV and PRV DataFrames
        hrv_values = df_hrv[metric].values
        prv_values = df_prv[metric].values  
        
        # Pearson (linear) correlation between the two sets of values
        pearson_corr, _ = pearsonr(prv_values, hrv_values)
        pearson_results[metric] = pearson_corr
        
        # Spearman (rank-based) correlation between the two sets of values
        spearman_corr, _ = spearmanr(prv_values, hrv_values)
        spearman_results[metric] = spearman_corr
    
    return pearson_results, spearman_results


# =============================================================================
# COMPUTE MEAN AND STD OF METRICS
# =============================================================================

def compute_mean_std_metrics(df, label):
    """
    Compute the mean and standard deviation for each row in a DataFrame (=all subjs)
    and return a formatted string of the form "mean ± std" for each metric.

    Parameters:
    - df (DataFrame): Input DataFrame where each row represents a metric 
                      and each column represents a different subject. 
    - label (str): Column label for the resulting DataFrame. 

    Returns:
    - formatted_df (DataFrame): DataFrame with one column containing 
                                formatted strings "mean ± std" for each metric.
    """

    # Compute mean and standard deviation along columns (per metric)
    mean_vals = df.mean(axis=1)
    std_vals = df.std(axis=1)

    # Define column name based on provided label
    col_name = label if label else "Value"

    # Combine mean and std into formatted strings (mean ± std)
    formatted_strings = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean_vals, std_vals)]
    formatted_df = pd.DataFrame({col_name: formatted_strings}, index=mean_vals.index)

    return formatted_df


# =============================================================================
# ORGANIZE DATA FOR COMPARISON (ALL WINDOWS vs METHOD 2)
# =============================================================================

def prepare_metric_dataframe(dict_all, dict_m2):
    """
    Prepares a structured DataFrame from two dictionaries of metric values.

    This function converts two dictionaries representing different conditions 
    ('All Windows' and 'Method 2') into a single DataFrame suitable for plotting. 
    Metric codes are mapped to readable titles, and the order is reversed for better 
    visualization (so the first metric appears at the top in plots).

    Parameters:
    - dict_all (dict): Dictionary containing values for the 'All Windows' condition.
    - dict_m2 (dict): Dictionary containing values for the 'Method 2' condition.
    - metric_to_title (dict): Mapping from metric codes to human-readable metric names.

    Returns:
    - pd.DataFrame: A DataFrame with columns ['Metric', 'All Windows', 'Method 2'] 
      and rows corresponding to metrics, ready for plotting.
    """

    df = pd.DataFrame({
        "Metric": list(dict_all.keys()),
        "All Windows": list(dict_all.values()),
        "Method 2": list(dict_m2.values())
    })

    # Map metric codes to readable titles
    df["Metric"] = df["Metric"].map(metric_to_title)

    return df


