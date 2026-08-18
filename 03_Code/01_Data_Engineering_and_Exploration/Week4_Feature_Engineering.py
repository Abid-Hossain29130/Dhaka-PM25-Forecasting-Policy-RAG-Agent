"""
================================================================================
  BLOCK 1: DATA MERGE & FEATURE ENGINEERING
  File: Week4_Feature_Engineering.py  (v4 — Final Data Alignment)
  Project: Dhaka AQI Research Paper
  Author: Abid Hossain (PI)

  Fixes applied in v4:
  1. Corrected AQI filename to 'Dhaka_AQI_Master_2022_2025.csv'
  2. Added AQI column renaming to match expected 'co', 'no2', etc.
  3. Added weather column renaming from raw open-meteo format
  4. Explicit timezone conversion for NASA files (UTC -> Asia/Dhaka)
  5. Updated expected daily row count to ~904.
================================================================================
"""

import pandas as pd
import numpy as np
import os

print("=" * 60)
print("  BLOCK 1: DATA MERGE & FEATURE ENGINEERING  [v4]")
print("=" * 60)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
RAW_DIR  = r"f:\Scholarship_Coach\Research_Hub\PQC_Readiness_Bangladesh\AQI_Paper\02_Data\01_Raw"
PROC_DIR = r"f:\Scholarship_Coach\Research_Hub\PQC_Readiness_Bangladesh\AQI_Paper\02_Data\02_Processed"
os.makedirs(PROC_DIR, exist_ok=True)

F_AQI  = os.path.join(PROC_DIR, "Dhaka_AQI_Cleaned.csv")   # Spatially filtered to Dhaka only
F_WTHR = os.path.join(RAW_DIR, "open-meteo-23.80N90.38E19m.csv")
F_AOD  = os.path.join(RAW_DIR, "NASA_Satellite_AOD_Extinction.csv")
F_SURF = os.path.join(RAW_DIR, "NASA_Satellite_Surface_PM25.csv")

# ---------------------------------------------------------
# BLOCK A: LOAD & PARSE (WITH COLUMN INSPECTION)
# ---------------------------------------------------------
print("\n[1/5] Loading and Parsing Raw Files...")

# 1. AQI Ground Data
df_aqi = pd.read_csv(F_AQI)
df_aqi.rename(columns={
    'carbon_monoxide': 'co',
    'nitrogen_dioxide': 'no2',
    'sulphur_dioxide': 'so2',
    'ozone': 'o3'
}, inplace=True)
df_aqi['datetime'] = pd.to_datetime(df_aqi['datetime'], utc=True).dt.tz_convert('Asia/Dhaka').dt.tz_localize(None)  # UTC → Dhaka local
n_stations = df_aqi['city_name'].nunique() if 'city_name' in df_aqi.columns else 'N/A'
print(f"  AQI loaded: {len(df_aqi):,} rows | Stations: {n_stations} | Columns: {df_aqi.columns.tolist()}")

# 2. Open-Meteo Weather (skip 2 metadata rows)
df_wthr = pd.read_csv(F_WTHR, skiprows=2)
# The raw Open-Meteo columns have units in them, e.g., 'windspeed_10m (km/h)'
df_wthr.columns = ['time', 'wind_speed_10m', 'wind_direction_10m', 'relative_humidity_2m', 'temperature_2m', 'precipitation', 'boundary_layer_height']
df_wthr['time'] = pd.to_datetime(df_wthr['time'])
df_wthr.rename(columns={'time': 'datetime'}, inplace=True)
print(f"  Weather loaded: {len(df_wthr):,} rows | Columns: {df_wthr.columns.tolist()}")

# 3. NASA AOD — auto-detect column name
df_aod = pd.read_csv(F_AOD, skiprows=8)
df_aod.columns = df_aod.columns.str.strip()
aod_val_col = [c for c in df_aod.columns if c != 'time'][0]
df_aod.rename(columns={'time': 'datetime', aod_val_col: 'aod_extinction'}, inplace=True)
# NASA data is UTC. Convert to Asia/Dhaka so the daily merge aligns correctly with local time
df_aod['datetime'] = pd.to_datetime(df_aod['datetime'], utc=True).dt.tz_convert('Asia/Dhaka').dt.tz_localize(None)
print(f"  AOD loaded: {len(df_aod):,} rows | Converted from UTC to Asia/Dhaka")

# 4. NASA Surface PM2.5 — same auto-detect approach
df_surf = pd.read_csv(F_SURF, skiprows=8)
df_surf.columns = df_surf.columns.str.strip()
surf_val_col = [c for c in df_surf.columns if c != 'time'][0]
df_surf.rename(columns={'time': 'datetime', surf_val_col: 'merra2_surf_pm25_kgm3'}, inplace=True)
# Convert from UTC to Asia/Dhaka
df_surf['datetime'] = pd.to_datetime(df_surf['datetime'], utc=True).dt.tz_convert('Asia/Dhaka').dt.tz_localize(None)
# Unit conversion: kg/m³ → µg/m³
df_surf['merra2_surf_pm25_ugm3'] = df_surf['merra2_surf_pm25_kgm3'] * 1e9
print(f"  Surface PM2.5 loaded: {len(df_surf):,} rows | Converted from UTC to Asia/Dhaka")

# ---------------------------------------------------------
# BLOCK B: HOURLY JOIN (MODEL A)
# ---------------------------------------------------------
print("\n[2/5] Building Hourly Master (Model A)...")

# Step 1: Spatial averaging — explicit numeric columns only
NUMERIC_COLS = ['pm2_5', 'pm10', 'no2', 'so2', 'o3', 'co', 'aqi']
df_aqi_mean = df_aqi.groupby('datetime')[NUMERIC_COLS].mean().reset_index()

# Step 2: Wind direction → U/V decomposition BEFORE daily aggregation
df_wthr['wind_u'] = df_wthr['wind_speed_10m'] * np.sin(np.deg2rad(df_wthr['wind_direction_10m']))
df_wthr['wind_v'] = df_wthr['wind_speed_10m'] * np.cos(np.deg2rad(df_wthr['wind_direction_10m']))

# Step 3: LEFT join
df_hourly = pd.merge(df_aqi_mean, df_wthr, on='datetime', how='left')

# Step 4: Cyclic hour encoding + lag features
df_hourly['hour'] = df_hourly['datetime'].dt.hour
df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly['hour'] / 24)
df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly['hour'] / 24)

df_hourly.sort_values('datetime', inplace=True)
df_hourly['pm25_lag1']  = df_hourly['pm2_5'].shift(1)
df_hourly['pm25_lag24'] = df_hourly['pm2_5'].shift(24)

# Filter to AQI data coverage period (data starts Aug 2022)
df_hourly = df_hourly[df_hourly['datetime'] >= pd.Timestamp('2022-08-01')]

# ---------------------------------------------------------
# BLOCK C: DAILY AGGREGATION (MODEL B)
# ---------------------------------------------------------
print("[3/5] Building Daily Master base (Model B)...")

df_hourly['date'] = df_hourly['datetime'].dt.date

# Daily ground aggregation
daily_ground = df_hourly.groupby('date').agg(
    pm2_5_mean        = ('pm2_5',                 'mean'),
    pm2_5_max         = ('pm2_5',                 'max'),
    wind_speed_mean   = ('wind_speed_10m',         'mean'),
    wind_u_mean       = ('wind_u',                'mean'),
    wind_v_mean       = ('wind_v',                'mean'),
    temperature_mean  = ('temperature_2m',         'mean'),
    rh_mean           = ('relative_humidity_2m',   'mean'),
    blh_mean          = ('boundary_layer_height',  'mean'),
    precip_sum        = ('precipitation',          'sum')
).reset_index()
daily_ground['date'] = pd.to_datetime(daily_ground['date'])

# Daily satellite aggregation
df_aod['date']  = df_aod['datetime'].dt.date
daily_aod = df_aod.groupby('date')['aod_extinction'].mean().reset_index()
daily_aod['date'] = pd.to_datetime(daily_aod['date'])

df_surf['date'] = df_surf['datetime'].dt.date
daily_surf = df_surf.groupby('date')['merra2_surf_pm25_ugm3'].mean().reset_index()
daily_surf['date'] = pd.to_datetime(daily_surf['date'])

# LEFT joins
df_daily = pd.merge(daily_ground, daily_aod,  on='date', how='left')
df_daily = pd.merge(df_daily,     daily_surf,  on='date', how='left')

# AOD missingness flag created HERE (after join)
df_daily['aod_missing'] = df_daily['aod_extinction'].isna().astype(int)

# Hard date cap: MERRA-2 data freeze
df_daily = df_daily[df_daily['date'] <= pd.Timestamp('2025-01-31')]

# Daily lag
df_daily.sort_values('date', inplace=True)
df_daily['pm25_lag7'] = df_daily['pm2_5_mean'].shift(7)

# ---------------------------------------------------------
# BLOCK D: SEASON ASSIGNMENT
# ---------------------------------------------------------
print("[4/5] Assigning Seasonal Regimes...")

def assign_season(dt_obj):
    m = dt_obj.month
    if m in [12, 1, 2]:    return 'Winter'
    if m in [3, 4, 5]:     return 'Pre_Monsoon'
    if m in [6, 7, 8, 9]:  return 'Monsoon'
    if m in [10, 11]:      return 'Post_Monsoon'

df_hourly['season'] = df_hourly['datetime'].apply(assign_season)
df_daily['season']  = df_daily['date'].apply(assign_season)

df_hourly.dropna(subset=['pm25_lag24'], inplace=True)
df_daily.dropna(subset=['pm25_lag7'],   inplace=True)

# ---------------------------------------------------------
# BLOCK E: VALIDATION REPORT
# ---------------------------------------------------------
print("\n" + "=" * 55)
print("  VALIDATION REPORT")
print("=" * 55)

print("\n[HOURLY MASTER - MODEL A]")
print(f"  Total rows        : {len(df_hourly):,}  (Expected ~21,000-28,900)")
print(f"  Date range        : {df_hourly['datetime'].min().date()} -> {df_hourly['datetime'].max().date()}")
print(f"  Missing PM2.5     : {df_hourly['pm2_5'].isna().sum()}")
print(f"  Weather gaps (BLH): {df_hourly['boundary_layer_height'].isna().sum()} rows")
print("  Season counts:")
print(df_hourly['season'].value_counts().to_string())

print("\n[DAILY MASTER - MODEL B]")
print(f"  Total rows        : {len(df_daily):,}  (Expected ~904 after lag7 drop)")
print(f"  Date range        : {df_daily['date'].min().date()} -> {df_daily['date'].max().date()}")
print(f"  AOD missing rows  : {df_daily['aod_missing'].sum()} (flagged, not dropped)")
print("  Season counts (All must be >= 100):")
print(df_daily['season'].value_counts().to_string())

# ---------------------------------------------------------
# BLOCK F: EXPORT
# ---------------------------------------------------------
print("\n[5/5] Exporting Dataframes...")
out_hourly = os.path.join(PROC_DIR, "master_hourly.csv")
out_daily  = os.path.join(PROC_DIR, "master_daily_base.csv")

df_hourly.to_csv(out_hourly, index=False)
df_daily.to_csv(out_daily,   index=False)

print(f"  OK Saved: {out_hourly}")
print(f"  OK Saved: {out_daily}")
print("=" * 60)
print("  BLOCK 1 COMPLETE - READY FOR BLOCK 2 (MODEL TRAINING)")
print("=" * 60)
