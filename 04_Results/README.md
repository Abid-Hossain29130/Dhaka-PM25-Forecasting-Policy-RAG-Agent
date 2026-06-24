# 📈 Results Directory (`04_Results/`)

## Overview
This directory contains all generated outputs from the PM₂.₅ forecasting pipeline and policy-auditing RAG agent. Results are organized by scenario and include model diagnostics, visualizations, policy briefs, and semantic evaluation metrics.

## 📁 Results Structure

### Scenario-Based Policy Briefs

**Environmental Action Advisory Briefs (EAABs)** are the primary output of the RAG policy agent. Each EAAB translates a PM₂.₅ forecast scenario into actionable policy guidance grounded in Bangladesh's statutory framework.

#### **1. Winter Inversion Scenario**
**File:** `EAAB_Winter_Inversion_Scenario.md`

- **Context:** December–February high-concentration episodes driven by atmospheric inversions and agricultural biomass burning
- **Trigger:** XGBoost forecast + SHAP analysis predicts PM₂.₅ > 200 μg/m³ for 24-hour ahead
- **Policy Content:**
  - Compliance gap analysis (vs. APCR 2022 24-h standard of 65 μg/m³)
  - Sectoral emission reduction targets (from NAQMP 2024-2030)
  - Health risk quantification (AF % by Thana from spatial mapping)
  - Specific interventions: Brick kiln regulation, transport diversions, industrial emission caps
  - Timeline: Immediate alerts + pre-positioning of contingency plans
- **Citations:** All policy references linked to `/02_Policy_Corpus/` with section anchors

#### **2. Monsoon Washout Scenario**
**File:** `EAAB_Monsoon_Washout_Scenario.md`

- **Context:** June–September low-concentration episodes due to atmospheric cleansing and wet deposition
- **Trigger:** XGBoost forecast predicts PM₂.₅ < 35 μg/m³ for 24-hour ahead
- **Policy Content:**
  - Achievement of WHO 2021 Interim Target-3 (35 μg/m³)
  - Maintenance recommendations for air quality monitoring networks
  - Seasonal planning for industrial production optimization
  - Public health messaging: Reduced health risk windows for vulnerable populations
  - Long-term strategy: Leverage monsoon as baseline for pollution source identification
- **Citations:** WHO 2021 Guidelines + NAQMP monitoring infrastructure chapters

#### **3. Extreme Episode Scenario**
**File:** `EAAB_Extreme_Episode_Scenario.md`

- **Context:** Episodic pollution spikes (wildfire smoke, industrial accidents, transboundary pollution)
- **Trigger:** XGBoost forecast + anomaly detection flags PM₂.₅ > 400 μg/m³ or Δ > 150 μg/m³ in 24 hours
- **Policy Content:**
  - Immediate emergency response activation per APCR 2022 Section 8 (Emergency Protocols)
  - Sector-specific shutdown orders (brick kilns, cement, steel plants)
  - Public health emergency declarations + hospital surge planning
  - Root cause diagnosis: Satellite data + meteorological analysis
  - Interagency coordination: DoE + Ministry of Health + Dhaka Metropolitan Police
- **Citations:** APCR 2022 Emergency Protocols + NAQMP Crisis Management sections

---

### Model Diagnostics & Evaluation

**`xgboost_model_metrics.json`**
- **Regression Metrics:**
  - R² (Test Set): 0.740
  - RMSE (μg/m³): 22.4
  - MAE (μg/m³): 16.8
  - MAPE (%): 18.2
- **Cross-Validation:** 5-fold temporal CV results (R² per fold)
- **Residual Diagnostics:** Mean residual = 0.8 μg/m³ (unbiased), Std = 21.2 μg/m³

**`shap_feature_importance.png`**
- Beeswarm plot of top 15 features ranked by |SHAP value|
- Top drivers: `PM25_lag1` (t-1 PM₂.₅), `temperature_mean_prev_day`, `wind_speed_mean_prev_day`
- Interpretation: Temporal persistence dominates; meteorological variables provide incremental value

**`test_predictions_vs_actuals.csv`**
- Columns: `date`, `actual_pm25`, `predicted_pm25`, `residual`, `shap_contribution_top5`
- 365 test samples (2024 calendar year)
- Enables direct validation of model calibration and scenario analysis

---

### Geospatial Outputs

**`thana_level_pm25_forecasts.geojson`**
- GeoJSON polygon features for all 92 Thanas
- Properties per Thana:
  - `pm25_forecast_mean` (μg/m³)
  - `pm25_forecast_std` (uncertainty quantile)
  - `attributable_fraction` (AF %)
  - `population_affected` (millions)
  - `health_burden_YLL` (Years of Life Lost, WHO methodology)

**`spatial_choropleth_pm25.html`**
- Interactive Folium map showing PM₂.₅ concentration by Thana
- Color scale: Green (< 35 WHO target) → Yellow (35–65 APCR standard) → Red (> 200 extreme)
- Hover tooltips display AF %, population, health burden per Thana

**`spatial_choropleth_health_impact.html`**
- Interactive Folium map showing Attributable Fraction (AF %) by Thana
- Highlights high-vulnerability districts (e.g., Dhaka Sadar, Mirpur with population density peaks)

---

### RAG Agent Evaluation

**`EAAB_eval_diagnostics.json`**
- Semantic evaluation metrics for each EAAB (using RAGAS-Proxy methodology):
  - **Faithfulness:** % policy statements grounded in ChromaDB retrieval (Target: > 95%)
  - **Answer Relevance:** Cosine similarity between query intent and EAAB content (Target: > 0.85)
  - **Context Precision:** % retrieved chunks that contribute to final answer (Target: > 0.80)
  - **Hallucination Ratio:** % claims not attributable to policy corpus (Target: < 5%)
  - **Citation Completeness:** % policy recommendations with source section links (Target: 100%)

**Example Output:**
```json
{
  "winter_scenario": {
    "faithfulness": 0.967,
    "answer_relevance": 0.89,
    "context_precision": 0.84,
    "hallucination_ratio": 0.02,
    "citation_completeness": 1.0
  }
}
```

**`retrieval_audit_log.json`**
- Detailed record of each RAG query:
  - Query text
  - Retrieved chunks (content + source policy + section)
  - Retrieval confidence (cosine similarity score)
  - Final policy recommendation derived from retrieval
- Enables manual auditing of agent behavior and cross-validation with policy experts

---

### Model & Agent Artifacts

**`xgboost_model_v1.pkl`**
- Serialized XGBoost regressor (trained model weights)
- Can be loaded via `joblib.load()` for inference on new data

**`chromadb_vectorstore/`**
- ChromaDB persistent storage of indexed policy corpus
- Contains three collections: `apcr_2022`, `naqmp_2024_2030`, `who_2021_guidelines`
- Enables reproducible RAG retrieval without re-embedding

---

## 📊 How to Interpret Results

### For Model Performance
1. Review `xgboost_model_metrics.json` for R² and residual bias
2. Cross-check `test_predictions_vs_actuals.csv` for outlier predictions
3. Examine `shap_feature_importance.png` to validate feature relevance

### For Spatial Health Impact
1. Open `spatial_choropleth_health_impact.html` in browser to visualize vulnerable zones
2. Cross-reference `thana_level_pm25_forecasts.geojson` for quantitative AF %
3. Identify high-risk Thanas (AF > 5%) for targeted interventions

### For Policy Grounding
1. Read one EAAB scenario (e.g., `EAAB_Winter_Inversion_Scenario.md`)
2. Cross-check cited policy sections in `/02_Policy_Corpus/Markdown_Embeddings/`
3. Review `EAAB_eval_diagnostics.json` to assess hallucination ratio and citation completeness
4. Inspect `retrieval_audit_log.json` to trace agent reasoning for each recommendation

---

## 🔄 Reproducibility & Version Control

- All results are reproducible from `/03_Code/` scripts with fixed random seeds
- Results are versioned by model timestamp: `results_v1_2026-06-24/`
- Key model hyperparameters logged in `model_config.json` for full auditability

---

## 📝 Usage Instructions for Reviewers

1. **Start with Model Validation:** `xgboost_model_metrics.json` + `shap_feature_importance.png`
2. **Assess Spatial Patterns:** `spatial_choropleth_pm25.html` + `thana_level_pm25_forecasts.geojson`
3. **Evaluate Policy Outputs:** Review all three EAAB Markdown files
4. **Audit RAG Grounding:** Cross-check citations in `retrieval_audit_log.json` against `/02_Policy_Corpus/`
5. **Validate Metrics:** `EAAB_eval_diagnostics.json` confirms hallucination mitigation and citation accuracy

---
**Last Updated:** June 2026  
**Results Generated By:** PM₂.₅ Forecasting Pipeline v1.0  
**Reproducibility:** 100% (fixed seeds, no external caching, offline evaluation)
