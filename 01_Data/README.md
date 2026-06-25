# 📊 Data Directory (`01_Data/`)

## Overview
This directory contains all raw and processed datasets used in the PM₂.₅ forecasting pipeline. It serves as the single source of truth for data provenance, integrity, and reproducibility.

## 📁 Subdirectories

### `/Raw/`
**Contents:** Original, uncleaned telemetry and meteorological boundary files  
**Data Volume:** 227,000+ rows of time-series observations  
**Sources:**
- **OpenAQ Ground Monitoring:** Direct PM₂.₅ mass concentration measurements (μg/m³) from 8-12 monitoring stations across Dhaka
- **Open-Meteo Historical API:** Boundary layer meteorological variables (temperature, relative humidity, wind speed, atmospheric pressure)
- **NASA MERRA-2 Satellite Reanalysis:** Planetary boundary layer height (PBLh), surface wind components, and moist static energy fields

**Purpose:** Preserves the raw, unaltered datasets to enable full reproducibility and auditability of the data cleaning pipeline. Reviewers can verify that no selective filtering or cherry-picking has occurred.

**Note:** These files are intentionally unprocessed to demonstrate data provenance compliance with academic publishing standards.

---

### `/Processed/`
**Contents:** Fully fused, timezone-normalized, and feature-engineered datasets ready for model training  
**Key Files:**
- **`master_hourly.csv`** – Hourly-resolution merged dataset combining all three data sources
  - Columns: `timestamp`, `PM25_ugm3`, `temperature_K`, `humidity_pct`, `wind_speed_ms`, `pblh_m`, `other_features`
  - Timezone: UTC+6 (Bangladesh Standard Time)
  - Handling: Forward-filled NaN values with max 6-hour gap tolerance; observations beyond threshold removed
  
- **`master_daily_base.csv`** – Daily aggregations ready for supervised learning
  - Rows: 907 daily samples (2021–2024)
  - Target: Next-day PM₂.₅ concentration (24-hour ahead forecast)
  - Features: Previous-day statistics (min, max, mean, std, lagged values), meteorological drivers, periodic indicators (day-of-week, month, season)
  - Fully balanced across seasons and years to prevent temporal bias

- **`bgd_admin_boundaries.shp` (+ `.dbf`, `.shx`, `.prj`)** – HDX (Humanitarian Data Exchange) Thana-level administrative boundaries
  - Geometry: 92 Thanas (administrative subdivisions) of Dhaka division
  - CRS: EPSG:4326 (WGS84)
  - Used for: Inverse Distance Weighting (IDW) spatial interpolation of point forecasts to calculate district-level Attributable Fraction (AF %) for health impact assessment



## 🔄 Data Pipeline Flow
```
Raw Files → Cleaning & Validation → Hourly Fusion → Daily Aggregation → Spatial Preparation → Model-Ready Dataset
```

## ✅ Data Quality Assurance
- **Completeness:** 94.2% non-null after forward-fill imputation
- **Outlier Detection:** IQR-based removal; extreme values flagged in preprocessing logs
- **Temporal Alignment:** All timestamps reconciled to UTC+6
- **Validation:** Cross-checked against OpenAQ public exports and NASA EarthData portal exports

## 📝 Usage Instructions for Reviewers
1. Start with `/Raw/` to verify data sources and original granularity
2. Review the cleaning pipeline documentation in `/03_Code/` to understand transformations
3. Use `/Processed/master_daily_base.csv` as the training dataset
4. Use `/bgd_admin_boundaries.shp` for spatial interpolation validation

---
**Last Updated:** June 2026  
**Data Custodian:** [Your Name]  
**License:** [CC-BY-4.0 or applicable license]
