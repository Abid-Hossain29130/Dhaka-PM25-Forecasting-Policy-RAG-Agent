# 💻 Code Vault (`03_Code/`)

## Overview
This directory contains the standalone, offline Python scripts that implement the complete PM₂.₅ forecasting and policy-auditing pipeline. These scripts follow strict MLOps principles: separation of concerns, modularity, and reproducibility. They are kept separate from the Colab notebook to enable version control, unit testing, and production deployment.

## 📁 Script Organization

### Data Engineering & Exploration

**`Week3_EDA.py`** – Exploratory Data Analysis  
- **Purpose:** Validate raw data sources, detect anomalies, and summarize distributional properties
- **Inputs:** Raw CSV files from `/01_Data/Raw/`
- **Outputs:** 
  - `eda_summary_statistics.json` (mean, std, quantiles, missing %)
  - `temporal_plots.png` (time series visualizations)
  - `correlation_matrix.png` (feature inter-dependencies)
- **Key Functions:**
  - `load_raw_data()` – Read and validate OpenAQ, Open-Meteo, MERRA-2 files
  - `detect_outliers()` – IQR-based anomaly flagging
  - `temporal_completeness()` – Calculate fill rates and gap lengths
  - `plot_distributions()` – Histograms and box plots per variable
- **Dependencies:** `pandas`, `numpy`, `matplotlib`, `seaborn`

**`Week4_Feature_Engineering.py`** – Data Fusion & Feature Construction  
- **Purpose:** Merge three data sources, normalize timestamps, and engineer time-series features
- **Inputs:** Cleaned Raw data + `bgd_admin_boundaries.shp` for spatial context
- **Outputs:**
  - `master_hourly.csv` (fully fused hourly dataset)
  - `master_daily_base.csv` (907 daily samples ready for ML)
  - `feature_engineering_log.json` (audit trail of transformations)
- **Key Functions:**
  - `merge_sources()` – Align OpenAQ, Open-Meteo, MERRA-2 by timestamp
  - `impute_missing()` – Forward-fill with 6-hour max gap threshold
  - `engineer_lags()` – Create t-1, t-7, t-365 lagged features
  - `engineer_cyclical()` – Hour-of-day, day-of-week, month (sine/cosine encoding)
  - `thana_spatial_context()` – Extract district-level metadata
- **Dependencies:** `pandas`, `geopandas`, `scipy.spatial`

---

### Model Training & Validation

**`Week5_Model_Training.py`** – XGBoost Forecasting Model  
- **Purpose:** Train and validate an XGBoost regressor for T+24h PM₂.₅ prediction with full hyperparameter tuning
- **Inputs:** `master_daily_base.csv`
- **Outputs:**
  - `xgboost_model.pkl` (trained model weights)
  - `training_metrics.json` (R², RMSE, MAE, MAPE)
  - `test_predictions.csv` (predictions vs. actuals)
  - `shap_values.pkl` (feature importance for interpretability)
- **Key Functions:**
  - `train_test_split()` – Temporal split (2021–2023 train, 2024 test)
  - `hyperparameter_tuning()` – GridSearchCV with 5-fold CV
  - `train_model()` – XGBoost with early stopping
  - `evaluate_model()` – Regression metrics + residual analysis
  - `compute_shap()` – SHAP beeswarm & dependence plots
- **Model Hyperparameters:**
  - `n_estimators: 200`, `max_depth: 6`, `learning_rate: 0.05`, `subsample: 0.8`
  - Early stopping on validation RMSE (patience=20)
- **Performance Benchmark:** R² = 0.740 on test set
- **Dependencies:** `xgboost`, `scikit-learn`, `shap`, `matplotlib`

---

### Geospatial Mapping & Health Risk

**`Week6_Spatial_Mapping_Collab_Version.py`** – IDW Interpolation & Health Impact  
- **Purpose:** Disaggregate point-level PM₂.₅ predictions to 92 Thanas via Inverse Distance Weighting; calculate health risk metrics (Attributable Fraction %)
- **Inputs:** 
  - `test_predictions.csv` (point forecasts at 8-12 monitoring stations)
  - `bgd_admin_boundaries.shp` (92 Thana polygons)
  - `health_exposure_matrix.csv` (population + baseline mortality per Thana)
- **Outputs:**
  - `thana_level_pm25_forecasts.geojson` (spatially disaggregated predictions)
  - `health_attributable_fraction.csv` (AF % per Thana using WHO exposure-response functions)
  - `spatial_maps.png` (choropleth visualizations of PM₂.₅ and AF %)
- **Key Functions:**
  - `idw_interpolation()` – Power-law IDW (power=2) across Thana centroids
  - `calculate_attributable_fraction()` – WHO health impact assessment methodology
  - `plot_choropleth()` – Interactive/static maps of spatial patterns
- **Dependencies:** `geopandas`, `shapely`, `folium`, `scipy`

---

### RAG Policy Agent

**`rag_run_full.py`** – Retrieval-Augmented Generation (RAG) Policy Agent  
- **Purpose:** Build a ChromaDB-backed LLM agent that translates PM₂.₅ forecasts into hallucination-free Environmental Action Advisory Briefs (EAABs)
- **Inputs:**
  - PM₂.₅ forecast from `test_predictions.csv`
  - Policy corpus (Markdown files from `/02_Policy_Corpus/Markdown_Embeddings/`)
  - Health impact data from `health_attributable_fraction.csv`
- **Outputs:**
  - `EAAB_Winter_Inversion_Scenario.md` – Winter policy brief
  - `EAAB_Monsoon_Washout_Scenario.md` – Monsoon policy brief
  - `EAAB_Extreme_Episode_Scenario.md` – Pollution episode brief
  - `EAAB_eval_diagnostics.json` – RAGAS-Proxy semantic evaluation metrics
- **Key Functions:**
  - `build_chromadb()` – Index Markdown policy files + embed with `text-embedding-3-small`
  - `initialize_rag_agent()` – LLM (GPT-4 or Claude) + retrieval chain
  - `generate_eaab()` – Multi-turn policy brief generation with grounding checks
  - `evaluate_eaab()` – Semantic similarity, hallucination detection, citation accuracy
- **RAG Architecture:**
  - **Retrieval:** Semantic similarity search (top-k=5 chunks)
  - **Grounding:** Cross-checks all policy citations against ChromaDB
  - **Hallucination Mitigation:** Constraint: "If policy not found, state 'insufficient guidance'" 
  - **Output Format:** Structured Markdown with inline policy citations
- **Dependencies:** `chromadb`, `langchain`, `openai`, `anthropic`

---

## 🔄 Execution Pipeline
```
Week3_EDA.py (Raw Data Validation)
    ↓
Week4_Feature_Engineering.py (Fusion & Feature Creation)
    ↓
Week5_Model_Training.py (XGBoost Training + SHAP)
    ↓
Week6_Spatial_Mapping_Collab_Version.py (IDW + Health Risk)
    ↓
rag_run_full.py (Policy Brief Generation & Evaluation)
```

## 🏃 How to Run Locally

### Prerequisites
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
export DHAKA_DATA_PATH="/path/to/01_Data/"
```

### Execution Steps
```bash
# Step 1: Exploratory Data Analysis
python Week3_EDA.py

# Step 2: Feature Engineering
python Week4_Feature_Engineering.py

# Step 3: Model Training
python Week5_Model_Training.py

# Step 4: Spatial Mapping
python Week6_Spatial_Mapping_Collab_Version.py

# Step 5: RAG Policy Agent
python rag_run_full.py
```

### Output Location
All outputs (CSVs, JSONs, plots, briefs) are saved to `/04_Results/`

## 🧪 Unit Testing
Each script includes internal validation checks:
- Data shape assertions post-merge
- Feature value range checks (e.g., PM₂.₅ ≥ 0)
- Model prediction bounds verification
- RAG retrieval relevance validation

## 📝 Usage Instructions for Reviewers
1. Review data engineering logic in `Week3_EDA.py` and `Week4_Feature_Engineering.py`
2. Validate model training reproducibility via `Week5_Model_Training.py` (fixed random seed)
3. Inspect spatial disaggregation in `Week6_Spatial_Mapping_Collab_Version.py`
4. Evaluate policy grounding in `rag_run_full.py` and cross-check `/04_Results/` outputs

## 📦 Dependencies
Full list available in `requirements.txt`:
- Data: `pandas>=1.5`, `geopandas>=0.12`, `numpy>=1.24`
- Modeling: `xgboost>=2.0`, `scikit-learn>=1.2`, `shap>=0.42`
- RAG: `chromadb>=0.3`, `langchain>=0.1`, `openai>=1.0`
- Viz: `matplotlib>=3.6`, `seaborn>=0.12`, `folium>=0.14`

---
**Last Updated:** June 2026  
**Code Maintainer:** [Your Name]  
**License:** [Your License Here]
