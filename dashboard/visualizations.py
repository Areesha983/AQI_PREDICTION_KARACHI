"""
Data dashboard visualization component engine.
Handles clean programmatic asset rendering for the Karachi AQI pipeline.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_error_progression(metrics_df: pd.DataFrame):
    """
    Generates the multi-horizon RMSE vs MAE progression line plot.
    Expects columns: 'Horizon', 'CV Mean RMSE', 'Test MAE'
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#121216')
    ax.set_facecolor('#121216')

    ax.plot(metrics_df["Horizon"], metrics_df["CV Mean RMSE"], marker='o', color='#ff4b4b', linewidth=2.5, label="RMSE Error Base")
    ax.plot(metrics_df["Horizon"], metrics_df["Test MAE"], marker='s', color='#ffa421', linewidth=2.5, label="MAE Deviation Base")

    ax.set_title("Prediction Error Progression Across Tracking Windows", color='white', fontsize=11, pad=12)
    ax.set_ylabel("Error Units (AQI Points)", color='#b0b3b8', fontsize=9)
    ax.set_xlabel("Forecast Horizon Scale", color='#b0b3b8', fontsize=9)

    ax.tick_params(colors='white', labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.15, color='gray')

    legend = ax.legend(facecolor='#1c1c24', edgecolor='none')
    for text in legend.get_texts():
        text.set_color('white')

    for spine in ax.spines.values():
        spine.set_color('#2d2d38')

    plt.tight_layout()
    return fig


def plot_variance_matrix(metrics_df: pd.DataFrame):
    """
    Generates the clean categorical variance bar chart.
    Expects columns: 'Horizon', 'Test R² Score'
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#121216')
    ax.set_facecolor('#121216')

    bars = ax.bar(metrics_df["Horizon"], metrics_df["Test R² Score"], color='#3b82f6', width=0.4, edgecolor='none')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

    ax.set_title("Variance Accounted For Matrix (R² Score Metric Profile)", color='white', fontsize=11, pad=12)
    ax.set_ylabel("R² Correlation Weight", color='#b0b3b8', fontsize=9)
    ax.set_xlabel("Target Horizon Gap", color='#b0b3b8', fontsize=9)

    ax.set_ylim(0, 0.6)
    ax.tick_params(colors='white', labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.15, color='gray')

    for spine in ax.spines.values():
        spine.set_color('#2d2d38')

    plt.tight_layout()
    return fig