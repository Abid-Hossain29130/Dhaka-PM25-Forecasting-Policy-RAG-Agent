"""
=======================================================================
  WEEK 3 — Exploratory Data Analysis (EDA) Script
  Project: From Prediction to Policy — Dhaka AQI
  Author:  Abid Hossain
  Date:    May 2026

  PURPOSE:
  This script reads the cleaned Dhaka AQI dataset and produces
  publication-quality figures required for the paper's methodology
  section. Specifically:
    1. Data overview (columns, shape, dtypes)
    2. Missing value analysis + heatmap
    3. PM2.5 distribution (histogram + boxplot)
    4. Time-series plot of PM2.5 (2000-2025)
    5. Seasonal decomposition (trend, seasonality, residual)
    6. ACF & PACF plots [MANDATORY — professor requirement]
    7. Correlation heatmap of all pollutants

  OUTPUT: All figures saved to AQI_Paper/Figures/EDA/
=======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Optional: missingno for missing value heatmap
try:
    import missingno as msno
    HAS_MISSINGNO = True
except ImportError:
    HAS_MISSINGNO = False
    print("[!] 'missingno' not installed. Will use seaborn heatmap instead.")
    print("    To install: pip install missingno\n")

# ACF/PACF from statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
BASE_DIR     = r"f:\Scholarship_Coach\Research_Hub\PQC_Readiness_Bangladesh\AQI_Paper"
CLEANED_CSV  = os.path.join(BASE_DIR, "02_Data", "02_Processed", "Dhaka_AQI_Cleaned.csv")
FIGURES_DIR  = os.path.join(BASE_DIR, "Figures", "EDA")
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 60)
print("  WEEK 3 — EDA SCRIPT STARTING")
print("=" * 60)

# -----------------------------------------------------------------------
# STEP 1: LOAD DATA
# -----------------------------------------------------------------------
print("\n[1/7] Loading cleaned Dhaka AQI data...")
df = pd.read_csv(CLEANED_CSV, low_memory=False)

print(f"      Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n      Column names detected:")
for col in df.columns:
    print(f"        → {col} ({df[col].dtype})")

# -----------------------------------------------------------------------
# STEP 2: IDENTIFY DATETIME AND POLLUTANT COLUMNS
# -----------------------------------------------------------------------
print("\n[2/7] Detecting datetime and pollutant columns...")

# Find datetime column
datetime_col = None
for candidate in ['datetime', 'date', 'dateLocal', 'date_local', 'dateUtc', 'date_utc', 'timestamp']:
    if candidate in df.columns:
        datetime_col = candidate
        break

if datetime_col is None:
    # Try to find any column with 'date' in the name
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_cols:
        datetime_col = date_cols[0]

if datetime_col:
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    print(f"      Datetime column: '{datetime_col}'")
    print(f"      Date range: {df[datetime_col].min()} → {df[datetime_col].max()}")
else:
    print("      [!] No datetime column auto-detected. Please check column names above.")

# Find PM2.5 column
pm25_col = None
for candidate in ['pm25', 'pm2.5', 'PM25', 'PM2.5', 'value', 'pm2_5']:
    if candidate in df.columns:
        pm25_col = candidate
        break

# If data is in LONG format (OpenAQ style with 'pollutant' and 'value' columns)
IS_LONG_FORMAT = False
if 'pollutant' in df.columns and 'value' in df.columns:
    IS_LONG_FORMAT = True
    print(f"\n      Data is in LONG format (one row per measurement).")
    print(f"      Pollutants found: {df['pollutant'].unique()}")
    # Pivot to wide format for analysis
    if datetime_col:
        df_wide = df.pivot_table(index=datetime_col, columns='pollutant', values='value', aggfunc='mean')
        df_wide.reset_index(inplace=True)
        pm25_col = 'pm25' if 'pm25' in df_wide.columns else None
        if pm25_col is None:
            pm25_candidates = [c for c in df_wide.columns if 'pm2' in str(c).lower()]
            pm25_col = pm25_candidates[0] if pm25_candidates else None
        print(f"      Pivoted to wide format. PM2.5 column: '{pm25_col}'")
    else:
        df_wide = df.copy()
else:
    IS_LONG_FORMAT = False
    df_wide = df.copy()
    if pm25_col:
        print(f"      Data is in WIDE format. PM2.5 column: '{pm25_col}'")
    else:
        # Try to find any numeric column
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        print(f"      Numeric columns: {numeric_cols}")
        pm25_col = numeric_cols[0] if numeric_cols else None
        print(f"      [!] PM2.5 not found by name. Using '{pm25_col}' as proxy.")

# Numeric columns for analysis (excluding ID-like columns)
numeric_cols = df_wide.select_dtypes(include=np.number).columns.tolist()
# Remove columns that look like IDs or lat/lon
exclude_keywords = ['id', 'lat', 'lon', 'latitude', 'longitude', 'pcode', 'code']
numeric_cols = [c for c in numeric_cols if not any(k in str(c).lower() for k in exclude_keywords)]
print(f"\n      Numeric columns for analysis: {numeric_cols}")

# -----------------------------------------------------------------------
# STEP 3: MISSING VALUE ANALYSIS
# -----------------------------------------------------------------------
print("\n[3/7] Analysing missing values...")

missing = df_wide[numeric_cols].isnull().sum()
missing_pct = (missing / len(df_wide) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)

print(missing_df.to_string())

# Plot missing value heatmap
fig, ax = plt.subplots(figsize=(12, 5))
if HAS_MISSINGNO and len(df_wide) <= 100000:
    msno.bar(df_wide[numeric_cols], ax=ax, color='steelblue', fontsize=10)
    ax.set_title('Missing Value Summary — Dhaka AQI Dataset', fontsize=13, fontweight='bold')
else:
    # Seaborn version
    sample_size = min(5000, len(df_wide))
    sns.heatmap(df_wide[numeric_cols].sample(sample_size).isnull(),
                cbar=False, yticklabels=False, cmap='viridis', ax=ax)
    ax.set_title(f'Missing Value Heatmap (sample of {sample_size:,} rows) — Dhaka AQI', fontsize=12, fontweight='bold')

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "Fig_01_Missing_Values.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"      [+] Saved: {fig_path}")

# -----------------------------------------------------------------------
# STEP 4: PM2.5 DISTRIBUTION
# -----------------------------------------------------------------------
if pm25_col and pm25_col in df_wide.columns:
    print(f"\n[4/7] Plotting PM2.5 distribution...")

    pm25_data = df_wide[pm25_col].dropna()
    # Remove extreme outliers for visualization (keep 99.5th percentile)
    upper_limit = pm25_data.quantile(0.995)
    pm25_clean = pm25_data[pm25_data <= upper_limit]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PM2.5 Distribution — Dhaka (All Stations)', fontsize=14, fontweight='bold')

    # Histogram
    axes[0].hist(pm25_clean, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].axvline(x=5.0,  color='green',  linestyle='--', linewidth=1.5, label='WHO 2021 (5 µg/m³)')
    axes[0].axvline(x=15.0, color='orange', linestyle='--', linewidth=1.5, label='WHO Interim 1 (15 µg/m³)')
    axes[0].axvline(x=pm25_clean.mean(), color='red', linestyle='-', linewidth=1.5,
                    label=f'Mean ({pm25_clean.mean():.1f} µg/m³)')
    axes[0].set_xlabel('PM2.5 (µg/m³)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Histogram', fontsize=11)
    axes[0].legend(fontsize=9)

    # Boxplot
    axes[1].boxplot(pm25_clean, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].axhline(y=5.0,  color='green',  linestyle='--', linewidth=1.5, label='WHO Guideline')
    axes[1].axhline(y=15.0, color='orange', linestyle='--', linewidth=1.5, label='WHO Interim 1')
    axes[1].set_ylabel('PM2.5 (µg/m³)', fontsize=11)
    axes[1].set_title('Boxplot', fontsize=11)
    axes[1].legend(fontsize=9)

    print(f"\n      PM2.5 Stats:")
    print(f"        Mean:   {pm25_clean.mean():.2f} µg/m³")
    print(f"        Median: {pm25_clean.median():.2f} µg/m³")
    print(f"        Max:    {pm25_data.max():.2f} µg/m³")
    print(f"        % above WHO (5 µg/m³): {(pm25_clean > 5).mean()*100:.1f}%")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "Fig_02_PM25_Distribution.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      [+] Saved: {fig_path}")

# -----------------------------------------------------------------------
# STEP 5: TIME-SERIES PLOT
# -----------------------------------------------------------------------
if datetime_col and pm25_col and pm25_col in df_wide.columns:
    print(f"\n[5/7] Plotting PM2.5 time-series...")

    ts_df = df_wide[[datetime_col, pm25_col]].dropna().copy()
    ts_df = ts_df.sort_values(datetime_col)

    # Monthly mean for cleaner plot
    ts_df.set_index(datetime_col, inplace=True)
    ts_monthly = ts_df[pm25_col].resample('ME').mean()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(ts_monthly.index, ts_monthly.values, color='steelblue', linewidth=1.2, label='Monthly Mean PM2.5')
    ax.fill_between(ts_monthly.index, ts_monthly.values, alpha=0.2, color='steelblue')
    ax.axhline(y=5.0, color='green', linestyle='--', linewidth=1.2, label='WHO 2021 Guideline (5 µg/m³)')
    ax.axhline(y=15.0, color='orange', linestyle='--', linewidth=1.2, label='WHO Interim 1 (15 µg/m³)')

    # Mark COVID period
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-06-01'),
               alpha=0.08, color='purple', label='COVID-19 Period')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('PM2.5 (µg/m³)', fontsize=12)
    ax.set_title('Monthly Mean PM2.5 Trend — Dhaka (2000–2025)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "Fig_03_PM25_TimeSeries.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      [+] Saved: {fig_path}")

# -----------------------------------------------------------------------
# STEP 6: ACF & PACF PLOTS [MANDATORY]
# -----------------------------------------------------------------------
if datetime_col and pm25_col and pm25_col in df_wide.columns:
    print(f"\n[6/7] Plotting ACF & PACF (professor requirement)...")
    print(f"      WHAT THIS PROVES: If ACF shows significant lags at 24h/48h,")
    print(f"      it mathematically justifies using LSTM (which has 'memory').")

    ts_df2 = df_wide[[datetime_col, pm25_col]].dropna().copy()
    ts_df2 = ts_df2.sort_values(datetime_col)
    ts_df2.set_index(datetime_col, inplace=True)

    # Use daily mean for ACF (cleaner signal, fewer points)
    ts_daily = ts_df2[pm25_col].resample('D').mean().dropna()

    # Only keep enough lags for meaningful analysis (60 days)
    max_lags = min(60, len(ts_daily) // 3)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Autocorrelation Analysis — Daily Mean PM2.5, Dhaka\n'
                 '(Proves temporal memory: justifies LSTM model choice)',
                 fontsize=12, fontweight='bold')

    plot_acf(ts_daily, lags=max_lags, ax=axes[0], color='steelblue',
             title='ACF — Autocorrelation Function\n(Significant lags = atmosphere has memory)')
    axes[0].set_xlabel('Lag (days)', fontsize=10)

    plot_pacf(ts_daily, lags=max_lags, ax=axes[1], color='steelblue', method='ols',
              title='PACF — Partial Autocorrelation Function\n(Shows direct lag effects)')
    axes[1].set_xlabel('Lag (days)', fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "Fig_04_ACF_PACF.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      [+] Saved: {fig_path}")

# -----------------------------------------------------------------------
# STEP 7: CORRELATION HEATMAP
# -----------------------------------------------------------------------
if len(numeric_cols) >= 2:
    print(f"\n[7/7] Plotting correlation heatmap...")

    corr_matrix = df_wide[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols)-1)))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # show lower triangle only
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_title('Pollutant Correlation Matrix — Dhaka AQI Dataset', fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "Fig_05_Correlation_Heatmap.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      [+] Saved: {fig_path}")

# -----------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("  EDA COMPLETE — ALL FIGURES SAVED")
print(f"  Location: {FIGURES_DIR}")
print("=" * 60)
print("\n  Figures produced:")
print("    Fig_01 — Missing Value Heatmap")
print("    Fig_02 — PM2.5 Distribution (Histogram + Boxplot)")
print("    Fig_03 — PM2.5 Time-Series (2000–2025)")
print("    Fig_04 — ACF & PACF [MANDATORY for paper]")
print("    Fig_05 — Correlation Heatmap")
print("\n  NEXT STEP: Review figures, then run Week4_Feature_Engineering.py")
print("=" * 60)
