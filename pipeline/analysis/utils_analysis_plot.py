"""
Code with functions to plot and visualize analysis and comparison between the 
various methods.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import math


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
# FUNCTION FOR VIOLIN PLOT FOR METRICS
# =============================================================================

def violin_plot(global_hrv_all, global_prv_all, global_hrv_m2, global_prv_m2):
    """
    Generates violin plots for HRV and PRV metrics across two conditions: 
    'All Windows' and 'Method 2'.

    Parameters:
    - global_hrv_all (DataFrame): HRV metrics (from ECG) computed over all windows.
    - global_prv_all (DataFrame): PRV metrics (from PPG) computed over all windows.
    - global_hrv_m2 (DataFrame): HRV metrics (from ECG) computed using Method 2.
    - global_prv_m2 (DataFrame): PRV metrics (from PPG) computed using Method 2.
    Each row of the dataframe corresponds to a specific metric.

    Returns:
    - None: Displays the violin plots for all metrics in a multi-panel figure.
    """

    colors={"ECG": "#E06C0E", "PPG": "#6282E3"}
    n_rows = math.ceil(len(all_metrics) / 4)
    fig, axes = plt.subplots(n_rows, 4, figsize=(25, 20))
    axes = axes.flatten()

    for i, metric in enumerate(all_metrics):
        ax = axes[i]

        # Values for All Windows
        hrv_values = global_hrv_all.loc[metric].values
        prv_values = global_prv_all.loc[metric].values

        # Values for Method 2
        hrv_values_m2 = global_hrv_m2.loc[metric].values
        prv_values_m2 = global_prv_m2.loc[metric].values

        # Prepare long-format DataFrame for plotting
        data = pd.DataFrame({
            "Value": list(hrv_values) + list(prv_values) + list(hrv_values_m2) + list(prv_values_m2),
            "Source": ["ECG"]*len(hrv_values) + ["PPG"]*len(prv_values) +
                      ["ECG"]*len(hrv_values_m2) + ["PPG"]*len(prv_values_m2),
            "Label": ["All Windows"]*(len(hrv_values) + len(prv_values)) +
                     ["M2"]*(len(hrv_values_m2) + len(prv_values_m2))
        })

        # Violin plot
        sns.violinplot(
            x="Label", y="Value", hue="Source",
            data=data, split=True, inner="quart", linewidth=1.2,
            ax=ax, palette=colors, legend=False
        )

        ax.set_title(metric_to_title[metric], fontsize=16, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=14)

    # Turn off unused axes
    for j in range(len(all_metrics), len(axes)):
        axes[j].axis('off')

    # Custom legend
    fig.legend(handles=[
        mpatches.Patch(color=colors["ECG"], label='ECG'),
        mpatches.Patch(color=colors["PPG"], label='PPG')
    ], loc='upper right', ncol=2, fontsize=20)

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    

# =============================================================================
# PLOT BARPLOT OF CORRELATIONS
# =============================================================================

def plot_correlations_bar(pearson_all, spearman_all, 
    pearson_m2, spearman_m2,
    pearson_all_std=None, spearman_all_std=None,
    pearson_m2_std=None, spearman_m2_std=None, 
    std_plot=False, title=""
    ):
    """
    Plots grouped bar charts comparing Pearson and Spearman correlation coefficients 
    between 'All Windows' and 'Method 2' conditions. Optionally includes standard deviation 
    error bars if `std_plot=True` (for global case).

    This function can be used in two ways:
    - Single-subject analysis: Provide only the correlation values for each metric 
      (set `std_plot=False`, standard deviation data not required).
    - Group-level analysis: Provide both mean and standard deviation values of correlations 
      computed across all subjects (set `std_plot=True` to include error bars).

    Parameters:
    - pearson_all (dict): Pearson correlations for the 'All Windows' condition.
    - spearman_all (dict): Spearman correlations for the 'All Windows' condition.
    - pearson_m2 (dict): Pearson correlations for the 'Method 2' condition.
    - spearman_m2 (dict): Spearman correlations for the 'Method 2' condition.
    - pearson_all_std (dict, optional): Standard deviation of Pearson correlations 
      for the 'All Windows' condition (used only if std_plot=True).
    - spearman_all_std (dict, optional): Standard deviation of Spearman correlations 
      for the 'All Windows' condition (used only if std_plot=True).
    - pearson_m2_std (dict, optional): Standard deviation of Pearson correlations 
      for the 'Method 2' condition (used only if std_plot=True).
    - spearman_m2_std (dict, optional): Standard deviation of Spearman correlations 
      for the 'Method 2' condition (used only if std_plot=True).
    - std_plot (bool, optional): If True, adds standard deviation error bars; 
      if False, displays numerical correlation values on the bars (default: False).
    - title (str, optional): Overall title of the figure (default: "").

    Returns:
    - None: Displays two bar plots showing Pearson and Spearman correlation coefficients 
      across metrics.
    """

    # Build DataFrames for plotting
    # All windows
    df_pearson_all = pd.DataFrame({"Metric": all_titles, "Pearson": list(pearson_all.values())})
    df_spearman_all = pd.DataFrame({"Metric": all_titles, "Spearman": list(spearman_all.values())})

    # Add Method 2
    cid = 'method 2'
    cid_name = cid.capitalize()
    df_pearson_all[f"Pearson {cid_name}"] = list(pearson_m2.values())
    df_spearman_all[f"Spearman {cid_name}"] = list(spearman_m2.values())


    fig, axes = plt.subplots(nrows=2, figsize=(24,14))
    fig.suptitle(title, fontsize=24, fontweight='bold')
    
    keys_pearson = ["Pearson", f"Pearson {cid_name}"]
    keys_spearman = ["Spearman", f"Spearman {cid_name}"]

    if std_plot:
        df_pearson_std = pd.DataFrame({"Metric": all_titles, "Pearson": list(pearson_all_std.values())})
        df_spearman_std = pd.DataFrame({"Metric": all_titles, "Spearman": list(spearman_all_std.values())})

        df_pearson_std[f"Pearson {cid_name}"] = list(pearson_m2_std.values())
        df_spearman_std[f"Spearman {cid_name}"] = list(spearman_m2_std.values())

        plot_bars_std(axes[0], df_pearson_all, df_pearson_std, keys_pearson, "Pearson Correlation")
        plot_bars_std(axes[1], df_spearman_all, df_spearman_std, keys_spearman, "Spearman Correlation")
    else:
        plot_bars(axes[0], df_pearson_all, keys_pearson, "Pearson Correlation")
        plot_bars(axes[1], df_spearman_all, keys_spearman, "Spearman Correlation")


    labels = ["All Windows", "M2"]
    fig.legend(labels, loc='upper right', bbox_to_anchor=(0.99, 1.07), fontsize=20, ncol=1)

    plt.tight_layout()
    plt.show()


# Helper function for plot_correlations_bar (to plot bars with std)
def plot_bars_std(ax, data, std_data, keys, ylabel):
    """
    Plot grouped bar charts with standard deviation error bars.

    Parameters:
    - ax (matplotlib.axes): Axis object where the bars will be plotted.
    - data (DataFrame): Mean values for each metric and condition.
    - std_data (DataFrame): Standard deviation values for each metric and condition.
    - keys (list of str): Columns in `data` and `std_data` to plot.
    - ylabel (str): Label for the Y-axis.
    """

    n_bars = len(keys)
    bar_width = 0.3 if n_bars == 2 else 0.25
    index = np.arange(len(data["Metric"]))  # X positions for metrics
    colors = ["#0F8A8C", "#BD216F"]
    positions = np.linspace(-bar_width * (n_bars - 1) / 2,
                            bar_width * (n_bars - 1) / 2,
                            n_bars)
    
    # Plot bars with standard deviation (error bars)
    for i, key in enumerate(keys):
        ax.bar(index + positions[i], data[key], bar_width, yerr=std_data[key],
               capsize=6, color=colors[i % len(colors)])
        
    # Axis formatting
    ax.set_ylabel(ylabel, fontsize=24, fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels(data["Metric"], rotation=45, ha='right', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    # Adjust horizontal limits for better layout
    min_pos = (index[0] + positions[0]) - bar_width / 2
    max_pos = (index[-1] + positions[-1]) + bar_width / 2
    ax.set_xlim(min_pos - 0.2, max_pos + 0.2)


# Helper function for plot_correlations_bar (to plot bars with numerical values)
def plot_bars(ax, data, keys, ylabel):
    """
    Plot grouped bar charts with annotated values.

    Parameters:
    - ax (matplotlib.axes): Axis object on which to plot the bars.
    - data (DataFrame): DataFrame containing metric names and their values.
    - keys (list of str): Column names in `data` representing groups to plot.
    - ylabel (str): Label for the Y-axis.
    """

    n_bars = len(keys)
    bar_width = 0.3 if n_bars == 2 else 0.25
    index = np.arange(len(data["Metric"]))  # X-axis positions
    colors = ["#0F8A8C", "#BD216F"]
    positions = np.linspace(-bar_width * (n_bars - 1) / 2,
                            bar_width * (n_bars - 1) / 2,
                            n_bars)

    # Plot bars and annotate values
    for i, key in enumerate(keys):
        bars = ax.bar(index + positions[i], data[key], bar_width,
                      color=colors[i % len(colors)])
        for b in bars:
            height = b.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(b.get_x() + b.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)

    # Axis formatting
    ax.set_ylabel(ylabel, fontsize=24, fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels(data["Metric"], rotation=45, ha='right', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    # Adjust horizontal limits for better layout
    min_pos = (index[0] + positions[0]) - bar_width / 2
    max_pos = (index[-1] + positions[-1]) + bar_width / 2
    ax.set_xlim(min_pos - 0.2, max_pos + 0.2)


# =============================================================================
# PLOT CV AND IQR HEATMAPS
# =============================================================================

def plot_heatmaps_comparison(df_pearson, df_spearman, title_prefix, cmap="coolwarm"):
    """
    General function to plot two side-by-side heatmaps (Pearson and Spearman)
    with a shared colorbar.

    Parameters:
    - df_pearson (DataFrame): DataFrame with columns ['Metric', 'All Windows', 'Method 2'].
    - df_spearman (DataFrame): DataFrame with columns ['Metric', 'All Windows', 'Method 2'].
    - title_prefix (str): Prefix for the subplot titles (e.g., "IQR" or "CV").
    - cmap (str): Matplotlib colormap to use (default: 'coolwarm').
    """

    # Pivot the DataFrames to wide format for heatmap plotting
    heatmap_pearson = df_pearson.set_index("Metric")[["All Windows", "Method 2"]]
    heatmap_spearman = df_spearman.set_index("Metric")[["All Windows", "Method 2"]]

    # Compute common vmin/vmax for shared color scale
    vmin = min(heatmap_pearson.min().min(), heatmap_spearman.min().min())
    vmax = max(heatmap_pearson.max().max(), heatmap_spearman.max().max())

    fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharey=True)

    plot_heatmap(heatmap_pearson, axes[0], f"{title_prefix} - Pearson Correlation", cmap, vmin, vmax)
    plot_heatmap(heatmap_spearman, axes[1], f"{title_prefix} - Spearman Correlation", cmap, vmin, vmax)

    # ColorBar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(axes[1].collections[0], cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


# Helper function for plot_heatmaps_comparison
def plot_heatmap(df, ax, title, cmap="coolwarm", vmin=None, vmax=None):
    """
    Plots a single heatmap for metric-based data (e.g., CV or IQR values).

    Parameters:
    - df (DataFrame): DataFrame containing the values to plot, 
      with metrics as rows and methods/conditions as columns.
    - ax (matplotlib.axes.Axes): Matplotlib axis object on which the heatmap will be drawn.
    - title (str): Title to display above the heatmap.
    - cmap (str, optional): Colormap used for visualization (default: 'coolwarm').
    - vmin (float, optional): Minimum value for color scaling. 
    - vmax (float, optional): Maximum value for color scaling.
    """

    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        ax=ax,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
        annot_kws={"size": 18}
    )
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xticklabels([label.get_text().title() for label in ax.get_xticklabels()], fontsize=16)
    ax.set_ylabel('')


# =============================================================================
# PLOT CV AND IQR LOLLIPOP
# =============================================================================

def plot_lollipop_comparison(df_pearson, df_spearman, title):
    """
    Creates a side-by-side lollipop plot comparison for Pearson and Spearman metrics.

    This function generates two subplots sharing the same x and y axes:
    - Left: Pearson correlations
    - Right: Spearman correlations
    Each subplot visualizes the difference between 'All Windows' and 'Method 2'.

    Parameters:
    - df_pearson (pd.DataFrame): DataFrame for Pearson metrics (columns: ['Metric', 'All Windows', 'Method 2']).
    - df_spearman (pd.DataFrame): DataFrame for Spearman metrics (columns: ['Metric', 'All Windows', 'Method 2']).
    - title (str): Common label for the x-axis and title reference for the figure.
    """
    
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(16, 9),
        sharex=True,
        sharey=True
    )

    # Reverse order so the first metric is plotted at the top
    df_pearson = df_pearson.iloc[::-1].reset_index(drop=True)
    df_spearman = df_spearman.iloc[::-1].reset_index(drop=True)

    # Plot each metric type
    plot_lollipop(axes[0], df_pearson, f"{title} - Pearson Correlation")
    plot_lollipop(axes[1], df_spearman, f"{title} - Spearman Correlation")

    # Set shared x-axis label
    axes[0].set_xlabel(title, fontsize=20)
    axes[1].set_xlabel(title, fontsize=20)

    # Unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2, fontsize=20)

    # Layout adjustments
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


# Helper function for plot_lollipop_comparison
def plot_lollipop(ax, df, title):
    """
    Draws a lollipop (dumbbell) plot on a given axis.

    This function visualizes the comparison between two conditions ('All Windows' 
    and 'Method 2') for each metric. Horizontal lines connect the two values, 
    and colored markers indicate each condition.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis object on which to draw the plot.
    - df (pd.DataFrame): DataFrame containing columns ['Metric', 'All Windows', 'Method 2'].
    - title (str): Title to display on the plot.
    """

    # Draw horizontal lines connecting the two values
    ax.hlines(
        y=df["Metric"],
        xmin=df["All Windows"],
        xmax=df["Method 2"],
        color='gray',
        alpha=0.6
    )

    # Plot All Windows points
    ax.scatter(
        df["All Windows"],
        df["Metric"],
        color="#0F8A8C",
        s=140,
        label="All Windows"
    )

    # Plot Method 2 points
    ax.scatter(
        df["Method 2"],
        df["Metric"],
        color="#BD216F",
        s=140,
        label="M2"
    )

    # Set title and axis styling
    ax.set_title(title, fontsize=22, fontweight='bold')
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)


# =============================================================================
# PLOT BOXPLOT CORRELATIONS 
# =============================================================================

def plot_boxplot_correlations(df_pearson_all, df_pearson_m2,
                              df_spearman_all, df_spearman_m2):
    """
    Plots two boxplots (Pearson and Spearman) showing the distribution of correlations
    across different HRV/PRV metrics and methods. 

    Parameters:
    - df_pearson_all / df_spearman_all: DataFrames in wide format (metrics as columns)
      for "All Windows".
    - df_pearson_m2 / df_spearman_m2: DataFrames in wide format for "Method 2".
    """

    # Convert all wide DataFrames to long format
    df_pearson_long = pd.concat([
        prepare_long_format(df_pearson_all, 'All Windows'),
        prepare_long_format(df_pearson_m2, 'Method 2')
    ])

    df_spearman_long = pd.concat([
        prepare_long_format(df_spearman_all, 'All Windows'),
        prepare_long_format(df_spearman_m2, 'Method 2')
    ])

    # Plotting
    fig, axes = plt.subplots(nrows=2, figsize=(24, 14))

    palette_colors = {
        "All Windows": "#0F8A8C",
        "Method 2": "#BD216F"
    }

    # Pearson
    sns.boxplot(data=df_pearson_long, x='Metric', y='Correlation', 
                hue='Method', ax=axes[0], palette=palette_colors)
    axes[0].set_ylabel('Pearson Correlation', fontsize=24, fontweight='bold')

    # Spearman
    sns.boxplot(data=df_spearman_long, x='Metric', y='Correlation', 
                hue='Method', ax=axes[1], palette=palette_colors)
    axes[1].set_ylabel('Spearman Correlation', fontsize=24, fontweight='bold')

    for ax in axes:
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45, labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.legend_.remove()

    custom_legend = [
        mpatches.Patch(color="#0F8A8C", label='All Windows'),
        mpatches.Patch(color="#BD216F", label='M2')
    ]
    fig.legend(handles=custom_legend, loc='upper left', bbox_to_anchor=(0.87, 1.07), fontsize=20, ncol=1)

    plt.tight_layout()
    plt.show()


# Helper function for plot_boxplot_correlations
def prepare_long_format(df, method_name):
    """
    Converts a wide-format DataFrame into long-format for plotting or analysis.

    Parameters:
    - df (pd.DataFrame): A DataFrame where columns are metrics and rows are observations.
    - method_name (str): Label to assign to the 'Method' column (All Windows or Method 2).

    Returns:
    - pd.DataFrame: A long-format DataFrame with columns:
        - 'Metric': the name of the metric (mapped to readable titles)
        - 'Correlation': the corresponding value
        - 'Method': the label identifying the experimental condition
    """
    
    # Melt the wide DataFrame into long format: each row corresponds to a single metric value
    df_long = df.melt(var_name='Metric', value_name='Correlation')
    
    # Map metric names to readable titles (using a predefined dictionary)
    df_long['Metric'] = df_long['Metric'].map(metric_to_title)
    
    # Add a column specifying the method/condition
    df_long['Method'] = method_name
    
    return df_long


# =============================================================================
# PLOT BLAND-ALTMAN
# =============================================================================

def plot_bland_altman_all(hrv_all, prv_all, hrv_m2, prv_m2):
    """
    Plot Bland-Altman plots for all HRV/PRV metrics under two conditions
    ('All Windows' and 'Method 2').

    Parameters:
    - hrv_all (dict): HRV metric values (All Windows), each key = metric, value = array of values.
    - prv_all (dict): PRV metric values (All Windows), each key = metric, value = array of values.
    - hrv_m2 (dict): HRV metric values (Method 2).
    - prv_m2 (dict): PRV metric values (Method 2).
    The data must be structured with metrics as columns and subjects as rows.
    """

    n_metrics = len(all_metrics)
    n_cols = 4
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 22))
    axes = axes.flatten()

    # Plot for each HRV/PRV metric under both conditions
    for i, metric in enumerate(all_metrics):
        plot_bland_altman(axes[i], np.array(hrv_all[metric]), np.array(prv_all[metric]),
                          "#0F8A8C", metric_to_title[metric])
        plot_bland_altman(axes[i], np.array(hrv_m2[metric]), np.array(prv_m2[metric]),
                          "#BD216F", metric_to_title[metric])

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.03, right=0.98, hspace=0.35, wspace=0.25)

    # Add legend
    handles = [
        mpatches.Patch(color=c, label=l)
        for c, l in [("#0F8A8C", "All Windows"), ("#BD216F", "M2")]
    ]
    #fig.legend(handles=handles, loc='upper right', ncol=1, fontsize=20, bbox_to_anchor=(0.98, 1.01))
    fig.legend(handles=handles, loc='upper right', ncol=1, fontsize=20, bbox_to_anchor=(0.99, 1.07))
    
    plt.tight_layout()
    plt.show()


# Helper function for plot_bland_altman_all
def bland_altman_stats(hrv_vals, prv_vals):
    """
    Compute Bland-Altman statistics for comparing two sets of measurements.

    Parameters:
    - hrv_vals (array-like): HRV metric values.
    - prv_vals (array-like): PRV metric values.

    Returns:
    - avg (ndarray): Average of HRV and PRV values for each observation.
    - diff (ndarray): Difference (HRV - PRV) for each observation.
    - mean_diff (float): Mean of the differences (bias).
    - loa_upper (float): Upper limit of agreement (mean_diff + 1.96 * SD).
    - loa_lower (float): Lower limit of agreement (mean_diff - 1.96 * SD).
    """

    avg = (hrv_vals + prv_vals) / 2
    diff = hrv_vals - prv_vals
    mean_diff = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_upper = mean_diff + 1.96 * sd
    loa_lower = mean_diff - 1.96 * sd
    return avg, diff, mean_diff, loa_upper, loa_lower


# Helper function for plot_bland_altman_all
def plot_bland_altman(ax, hrv_vals, prv_vals, color, title):
    """
    Plot a single Bland-Altman plot comparing HRV and PRV metric.

    Parameters:
    - ax (matplotlib.axes.Axes): Axis to draw the plot on.
    - hrv_vals (array-like): HRV metric values.
    - prv_vals (array-like): PRV metric values.
    - color (str): Color for the plot points and lines.
    - title (str): Title for the subplot.
    """

    avg, diff, mean_diff, loa_upper, loa_lower = bland_altman_stats(hrv_vals, prv_vals)
    ax.scatter(avg, diff, color=color, s=80)
    ax.axhline(mean_diff, color=color, linestyle="-", linewidth=2.5)
    ax.axhline(loa_upper, color=color, linestyle="--", linewidth=2.3)
    ax.axhline(loa_lower, color=color, linestyle="--", linewidth=2.3)
    ax.set_title(title, fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)


# =============================================================================
# PLOT GROUPED BARPLOT CORRELATIONS (healthy, RBD, OSAS, PLM, mixed)
# =============================================================================

def plot_grouped_bars(ax, data_mean, data_std, keys, ylabel):
    """
    Plots grouped bar charts for multiple groups across a set of HRV/PRVmetrics.

    Parameters:
    - ax (matplotlib.axes): The axis object on which to plot the bars.
    - data_mean (pandas.DataFrame): A DataFrame containing the mean of correlation values.
    - data_std (pandas.DataFrame): A DataFrame containing the std of correlation values.
    - keys (list of str): The column names in `data` corresponding to the groups 
      (e.g., 'healthy', 'OSAS', etc.).
    - ylabel (str): The label for the Y-axis.
    """
    
    n_bars = len(keys)  # Number of bars per group
    bar_width = 0.16    
    index = np.arange(len(data_mean["Metric"]))  # X positions for the metrics
    colors = ['green', 'darkorange', 'gold', 'skyblue', 'mediumpurple']  
    
    # Compute the bar positions for each group
    positions = np.linspace(-bar_width * (n_bars-1) / 2,
                            bar_width * (n_bars-1) / 2,
                            n_bars)
    
    # Plot bars for each group  
    for i, key in enumerate(keys):
        bars = ax.bar(
                index + positions[i],
                data_mean[key],
                bar_width,
                color=colors[i % len(colors)],
                yerr=data_std[key], 
                capsize=4,      
                ecolor='black'
        )
        # Add value labels on top of each bar
        # for b in bars:
        #    height = b.get_height()
        #    ax.annotate(f'{height:.2f}', 
        #                xy=(b.get_x() + b.get_width() / 2, height),
        #                xytext=(0, 5), textcoords="offset points",
        #                ha='center', va='bottom', fontsize=10)
    
    # Axis formatting    
    ax.set_ylabel(ylabel, fontsize=24, fontweight='bold')
    ax.set_xticks(index)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xticklabels(all_titles, rotation=45, ha='right', fontsize=20) 
    
    # Remove extra white space from the beginning and end of subplot
    min_pos = (index[0] + positions[0]) - bar_width / 2
    max_pos = (index[-1] + positions[-1]) + bar_width / 2
    ax.set_xlim(min_pos - 0.2, max_pos + 0.2)


# =============================================================================
# PLOT HRV/PRV METRIC OVER WINDOWS
# =============================================================================

def plot_metrics(df_hrv, df_prv, pearson, spearman, subj, group, name_method):
    """
    Plots HRV and PRV metrics for a single subject, showing their temporal trends
    and the corresponding correlation coefficients.

    Each subplot displays one metric, comparing HRV and PRV features over time.
    The Pearson (r) and Spearman (ρ) correlation coefficients for each metric 
    are shown in the subplot titles.

    Parameters:
    - df_hrv (DataFrame): HRV metrics for ECG, with each row corresponding to a window.
    - df_prv (DataFrame): PRV metrics for PPG, with each row corresponding to a window.
    - pearson (dict): Values for Pearson correlations.
    - spearman (dict): Values for Spearman correlations.
    - subj (str): Subject identifier.
    - group (str): Group to which the subject belongs (e.g., 'healthy', 'RBD', etc.).
    - name_method (str): Processing method used ('All Windows' or 'Method 2').
    """

    n_rows = math.ceil(len(all_metrics)/4)
    fig, axes = plt.subplots(n_rows, 4, figsize=(25,20))
    fig.suptitle(f"{subj} ({group}) - {name_method}", fontsize=22, fontweight='bold')
    axes = axes.flatten()

    for i, (metric, title) in enumerate(zip(all_metrics, all_titles)):
        ax = axes[i]
        ax.plot(df_hrv[metric], label="HRV Features", color='darkorange', linewidth=3)
        ax.plot(df_prv[metric], label="PRV Features", color='blue', linewidth=2.4)
        ax.set_title(f"{title} (r={pearson[metric]:.2f}, ρ={spearman[metric]:.2f})", 
                        fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xticklabels([])

    for j in range(len(all_metrics), len(axes)):
        axes[j].axis('off')
        empty_ax = axes[j]

    handles, labels = ax.get_legend_handles_labels()
    empty_ax.legend(handles, labels, loc='center left', fontsize=24, ncol=1, frameon=False)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()