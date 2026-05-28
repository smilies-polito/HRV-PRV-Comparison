"""
Code with functions to perform statistical and equivalence tests.
"""


import pandas as pd
from scipy.stats import shapiro, wilcoxon, norm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pingouin as pg
from matplotlib.colors import ListedColormap


# Metrics of interest for analysis
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
# FUNCTIONS FOR STATISTICAL TEST
# =============================================================================

def check_normality(df_hrv, df_prv):
    """
    Checks the normality of each shared feature between HRV and PRV dataframes
    using the Shapiro-Wilk test.

    Parameters:
    - df_hrv (DataFrame): HRV metrics for ECG, with each row corresponding to a window.
    - df_prv (DataFrame): PRV metrics for PPG, with each row corresponding to a window.

    Returns:
    - results (list of tuples): Each tuple contains (feature_name, is_normal), where `is_normal` is True
        if both HRV and PRV distributions are normally distributed (p > 0.05).
    """

    results = []

    for metric in all_metrics:
        hrv_values = df_hrv[metric].dropna()
        prv_values = df_prv[metric].dropna()

        # Perform Shapiro–Wilk normality test
        _, hrv_p_value = shapiro(hrv_values)
        _, prv_p_value = shapiro(prv_values)

        # Normal only if p > 0.05
        # A feature is considered normal only if both HRV and PRV are normally distributed
        is_normal = np.logical_and(hrv_p_value > 0.05, prv_p_value > 0.05)
        results.append((metric, is_normal))

    return results



def test_wilcoxon(df_hrv, df_prv):
    """
    Performs statistical comparison (Wilcoxon) of HRV and PRV features for each metric.

    Parameters:
    - df_hrv (DataFrame): HRV metrics for ECG, with each row corresponding to a window.
    - df_prv (DataFrame): PRV metrics for PPG, with each row corresponding to a window.

    Returns:
    - results (pd.DataFrame): Containing the following columns for each feature:
        - 'Feature': Feature name
        - 'p_value': p-value of the test
        - 'Significant': True if p < 0.05
        - 'Effect size': Effect size 
    """

    results = []

    for metric in all_metrics:
        hrv_values = df_hrv[metric].dropna()
        prv_values = df_prv[metric].dropna()

        # Skip features with insufficient paired data
        if len(hrv_values) != len(prv_values) or len(hrv_values) < 3:
            continue

        diff = hrv_values - prv_values

        # Wilcoxon signed-rank test for non-normal data
        _, p_value = wilcoxon(hrv_values, prv_values)
        z = norm.ppf(p_value / 2) * -1  # Approximate z-score
        N = np.sum(diff != 0)
        effect_size = z / np.sqrt(N) if N > 0 else np.nan

        p_value = round(p_value, 5)
        effect_size = round(effect_size, 5) if not np.isnan(effect_size) else np.nan

        # Store test results
        results.append({
            'Feature': metric,
            'p_value': p_value,
            'Significant': p_value < 0.05,
            'Effect size': effect_size
        })

    return pd.DataFrame(results)



def plot_pvalue_heatmap(df, method_name):
    """
    Create a heatmap showing p-values for each feature-subject pair.
    - Yellow cells: non-significant differences (p > 0.05)
    - White cells: significant differences (p <= 0.05)
    - p-values are annotated in each cell

    Parameters:
    - df (pd.DataFrame): Matrix of p-values (features x subjects)
    - method_name (str): Title indicating the method being visualized
    """

    # Binary mask: True where p <= 0.05 (significant)
    significant_mask = df <= 0.05

    # Map: 0 -> non-significant (yellow), 1 -> significant (white)
    color_data = significant_mask.astype(int)
    cmap = ListedColormap(['yellow', 'white'])
    df = df.rename(index=metric_to_title)

    plt.figure(figsize=(max(7, 1.1 * df.shape[1]), max(6, 0.4 * df.shape[0])))

    ax = sns.heatmap(
        color_data,
        cmap=cmap,
        cbar=False,
        annot=df,
        fmt=".2f",
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"size": 6, "color": "black"},
        square=False
    )

    # Formatting
    plt.title(f'Statistical Significance (p-values) - {method_name}', fontsize=14, pad=15)
    plt.xlabel('Subject', fontsize=9)
    plt.ylabel('HRV Feature', fontsize=9)
    plt.xticks(rotation=45,fontsize=6, ha='right')
    plt.yticks(rotation=0, fontsize=5, ha='right')
    plt.tight_layout(pad=2)
    plt.show()
    
    

# =============================================================================
# FUNCTIONS FOR EQUIVALENCE TEST
# =============================================================================

def compute_median_iqr(values_dict):
    """
    Compute the global median and interquartile range (IQR) for each feature.

    Parameter:
    - values_dict (dict): A dictionary where keys are feature names and values
                          are lists of numerical values collected across all subjects.

    Returns:
    - medians (dict): Dictionary mapping each feature to its global median.
    - iqrs (dict): Dictionary mapping each feature to its global IQR.
    """

    medians, iqrs = {}, {}
    for col, values in values_dict.items():
        arr = np.array(values)
        medians[col] = np.nanmedian(arr)
        iqrs[col] = np.nanquantile(arr, 0.75) - np.nanquantile(arr, 0.25)
    return medians, iqrs



def robust_scale(df, medians, iqrs): 
    """
    Apply robust scaling to all numeric features in a DataFrame.
    Each feature is standardized using its median and interquartile range (IQR):
        scaled_value = (value - median) / IQR

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing numeric features to scale.
    - medians (dict): Dictionary mapping each feature to its global median.
    - iqrs (dict): Dictionary mapping each feature to its global IQR.

    Returns:
    - pd.DataFrame: A new DataFrame with robustly scaled feature values.
    """

    df_scaled = df.copy()
    for col in df.columns:
        df_scaled[col] = (df[col] - medians[col]) / iqrs[col]
    return df_scaled



def test_equivalence_tost(df_hrv, df_prv, epsilon):
    """
    Perform a paired Two One-Sided Test (TOST) for each feature between HRV and PRV.

    Parameters:
    - df_hrv (DataFrame): HRV metrics for ECG, with each row corresponding to a window.
    - df_prv (DataFrame): PRV metrics for PPG, with each row corresponding to a window.
    - epsilon (float): equivalence bound.

    Returns:
    - pd.DataFrame of summary table with feature name, p-value, bound, degrees of freedom,
        and whether the two signals are equivalent.
    """

    results = []

    for metric in all_metrics:
        x = df_hrv[metric].values
        y = df_prv[metric].values

        # Run paired TOST
        tost_result = pg.tost(x=x, y=y, bound=epsilon, paired=True)

        pval = tost_result['pval'].values[0]
        equivalent = pval < 0.05  # significant => equivalent

        results.append({
            'Feature': metric,
            'p_value': pval,
            'epsilon': epsilon,
            'dof': tost_result['dof'].values[0],
            'Equivalent': equivalent
        })

    return pd.DataFrame(results)



def plot_equivalence_heatmap(df_equiv, method_name, epsilon):
    """
    Plot a heatmap showing feature-wise equivalence results (HRV vs PRV) across subjects.
    Each cell represents whether a given feature is statistically equivalent 
    between HRV and PRV for a specific subject, according to the TOST test.
    Green = equivalent, Red = not equivalent.

    Parameters:
    - df_equiv (DataFrame): Boolean values (True = equivalent, False = not equivalent), 
                            where rows correspond to features and columns correspond to subjects.
    - method_name (str): Name of the analysis method ("All Windows" or "Method 2").
    - epsilon (float): Equivalence bound (ε) used in the TOST test.
    """

    df_equiv = df_equiv.rename(index=metric_to_title)
    plt.figure(figsize=(14, 6))
    
    sns.heatmap(
        df_equiv.astype(bool),  
        annot=True,             
        cmap=sns.color_palette(['lightcoral', 'lightgreen']),  
        cbar=False,             
        fmt='',                 
        linewidths=0.5,         
        linecolor='gray',       
        annot_kws={"size": 8}   
    )
    
    plt.title(f"Equivalence Test (ε = {epsilon}) - {method_name}", fontsize=12)
    plt.xlabel(f"Subjects", fontsize=10)
    plt.ylabel(f"Features", fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, ha='right', fontsize=8)
    plt.tight_layout()
    plt.show()
