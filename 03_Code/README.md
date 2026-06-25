# 💻 Codebase & MLOps Pipeline (`/03_Code`)

## 📌 Overview
This directory contains the core Python codebase that powers the Dhaka PM₂.₅ forecasting model, the spatial health risk interpolator, and the RAG-based LLM policy agent. 

To maintain strict MLOps principles, the heavy offline data engineering (ETL) and modular component scripts are kept here. This keeps the primary `MASTER_IMPLEMENTATION_DHAKA_PM25.ipynb` (in the root directory) lightweight and optimized for fast execution and demonstration.

---

## 📂 Directory Guide

### 1️⃣ `Data Engineering & Exploration/`
This module handles the extraction, transformation, and loading (ETL) of multi-source environmental data, fusing ground telemetry, weather APIs, and satellite reanalysis into a single target-ready dataset.
*   **`Week3_EDA.py`**: Performs Exploratory Data Analysis (EDA) on the raw OpenAQ telemetry. It generates statistical summaries, handles temporal completeness checks, and plots initial distributions (ACF/PACF, correlation heatmaps).
*   **`Week4_Feature_Engineering.py`**: Fuses the ground data with Open-Meteo boundary layer meteorology and NASA MERRA-2 satellite AOD. It manages timezone normalization (UTC to BDT), wind vector decomposition (U/V), and the engineering of deterministic diurnal micro-physics features.

### 2️⃣ `Model Training & Validation/`
This module contains the core predictive machine learning algorithms and statistical rigor tests.
*   **`Week5_Model_Training.py`**: Trains the XGBoost forecasting model to predict next-day (T+24h) PM₂.₅ concentrations. 
    *   **Validation:** Uses strict chronological `TimeSeriesSplit(n_splits=5)` to eliminate look-ahead bias.
    *   **Explainability (XAI):** Implements `shap.TreeExplainer` to extract physical feature drivers (e.g., boundary layer height vs. aerosol loading).
    *   **Diagnostics:** Runs Diebold-Mariano and Ljung-Box statistical tests on the residuals.

### 3️⃣ `Geospatial Mapping & Health Risk/`
This module bridges the gap between atmospheric predictions and epidemiological public health risks.
*   **`Week6_Spatial_Mapping_Collab_Version.py`**: Ingests the singular city-wide PM₂.₅ forecast and applies Inverse Distance Weighting (IDW) to disaggregate the exposure risk across 92 administrative Thanas (districts) in Dhaka. It calculates the Attributable Fraction (AF%) of health risk using established Concentration-Response Functions (CRFs) and generates interactive `folium` choropleth maps.

### 4️⃣ `RAG Policy Agent/`
This module translates the numerical forecasts and physical SHAP drivers into actionable, hallucination-free urban governance directives.
*   **`rag_run_full.py` & `RAG_RUN_FULL.ipynb`**: The implementation of the Policy-Auditing Retrieval-Augmented Generation (RAG) Agent.
    *   **Vector Database:** Embeds the Bangladesh Air Pollution Control Rules (APCR 2022), the National Air Quality Management Plan (NAQMP), and WHO Guidelines using `all-MiniLM-L6-v2` into a local ChromaDB instance.
    *   **LLM Orchestration:** Uses the Groq API (`openai/gpt-oss-120b`) to synthesize the retrieved legal chunks and forecasted data into Environmental Action Advisory Briefs (EAABs).
    *   **Evaluation:** Includes a local RAGAS-Proxy diagnostic runner to evaluate semantic faithfulness, context precision, and hallucination rates.

---

## ⚙️ Usage Note
While these standalone scripts represent the complete pipeline, **reviewers and visitors are encouraged to run the `MASTER_IMPLEMENTATION_DHAKA_PM25.ipynb` file located in the root directory**. That notebook serves as a compiled "Execution Engine" that orchestrates these modules seamlessly for rapid demonstration.
