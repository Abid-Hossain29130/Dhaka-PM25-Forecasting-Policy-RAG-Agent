# 📊 Data Directory (`/01_Data`)

## 📌 Overview
This directory contains all raw and processed datasets used in the PM₂.₅ forecasting and geospatial health risk pipeline. It serves as the single source of truth for data provenance, integrity, and reproducibility.

---

## 📁 Directory Guide

### 1️⃣ `/Raw/`
**Contents:** Original, uncleaned telemetry, meteorological boundary files, and geospatial administrative boundaries.
**Data Volume:** 227,000+ rows of initial time-series observations.

**Authoritative Sources:**
- **OpenAQ Ground Monitoring:** Direct PM₂.₅ mass concentration measurements (μg/m³) from 3 active monitoring stations across Dhaka (e.g., US Embassy Baridhara, DoE Motijheel, DoE Gulshan).
- **Open-Meteo Historical API:** Boundary layer meteorological variables (temperature, relative humidity, wind speed, wind direction, precipitation, boundary layer height).
- **NASA MERRA-2 Satellite Reanalysis:** Aerosol Optical Depth (AOD) extinction at 550 nm and satellite-simulated surface PM₂.₅ estimates.
- **HDX & BBS Census:** Raw administrative shapefiles and population data.

**Purpose:** Preserves the raw, unaltered datasets to enable full reproducibility of the data fusion pipeline. Reviewers can verify that no selective filtering or cherry-picking has occurred prior to programmatic cleaning.

### 2️⃣ `/Processed/`
**Contents:** Fully fused, timezone-normalized, and feature-engineered datasets ready for model training and spatial interpolation.

**Key Files:**
- **`master_hourly.csv`** – Hourly-resolution merged dataset.
  * **Rows:** 21,937 valid hourly records.
  * **Timezone:** Normalized from UTC to UTC+6 (Bangladesh Standard Time) to align diurnal emissions with local meteorology.
  * **Quality Control:** Pruned exclusively using physical sensor constraints (values <0 or >999 μg/m³ removed). 
- **`master_daily_base.csv`** – Daily aggregations ready for supervised learning.
  * **Rows:** 907 target-ready daily samples spanning August 1, 2022, to January 31, 2025.
  * **Target:** Next-day PM₂.₅ concentration (T+24h forecast) generated via rigorous `shift(-1)` alignment.
  * **Features:** 25 optimized predictors including autoregressive lags, meteorological boundary conditions, satellite aerosol loading, and deterministic diurnal micro-physics features.
- **`bgd_admin_boundaries.shp` (alongside `.dbf`, `.shx`, `.prj`)** – HDX Thana-level administrative boundaries.
  * **Geometry:** 92 administrative Thanas of the Dhaka district.
  * **Purpose:** Used for Inverse Distance Weighting (IDW) spatial interpolation to calculate localized Attributable Fraction (AF%) health risks.

---

## ✅ Data Quality Assurance & MLOps Strictures
To prevent the artificial inflation of predictive accuracy (a common flaw in air quality literature), strict data-cleansing parameters were enforced:
- **Completeness Threshold:** A strict ≥75% daily completeness threshold (minimum 18 valid hours per day) was enforced to prevent uncertainty inflation when calculating daily statistics.
- **Outlier Handling (No IQR):** No statistical IQR dropping was performed in order to preserve real-world extreme inversion episodes. Only physically impossible sensor artifacts were pruned.
- **Leakage-Free Imputation:** No forward-fill imputation was applied globally. XGBoost natively handles sparse data, while baseline models utilized strict fold-internal median imputation during chronological validation to prevent temporal leakage.
- **Contiguity:** The final array was programmatically asserted to cover 908 consecutive days with zero gaps.

## 📝 Usage Instructions for Reviewers
1. Start with the `/Raw/` folder to verify initial data granularity.
2. Review the data engineering logic in `/03_Code/Data Engineering & Exploration/` to trace the ETL process.
3. Use `/01_Data/Processed/master_daily_base.csv` as the explicit target matrix for reproducing the $R^2 = 0.740$ benchmark.
4. Call upon the `/Processed/` shapefiles to replicate the Folium choropleth mapping.

---
**License:** MIT License
