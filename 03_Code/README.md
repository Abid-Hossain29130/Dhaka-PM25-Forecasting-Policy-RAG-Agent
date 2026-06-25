# 💻 Code Vault (`03_Code/`)

## 📌 Overview
This directory serves as the "Code Vault" for the offline, standalone Python scripts that built the PM₂.₅ forecasting, spatial health risk, and policy-auditing pipeline. 

In accordance with strict MLOps (Machine Learning Operations) principles, the heavy data engineering (ETL) and modular component testing scripts are kept in this folder. This prevents the primary `MASTER_IMPLEMENTATION_DHAKA_PM25.ipynb` execution engine (located in the root directory) from experiencing memory overloads or browser crashes when running the final presentation for reviewers.

---

## 📂 Canonical Script Organization

### Block 1: Multi-Source Data Fusion Pipeline
These scripts handle the offline data ingestion, timezone normalization, and feature creation from ground sensors, meteorological models, and satellite reanalysis.

*   **`Week3_EDA.py`**
    *   **Purpose:** Executes Exploratory Data Analysis (EDA) on the raw telemetry. Generates the visual and statistical justifications used in Chapter 3, including PM₂.₅ distributions, temporal completeness, ACF/PACF plots, and the correlation heatmap.
*   **`Week4_Feature_Engineering.py`**
    *   **Purpose:** Fuses OpenAQ ground measurements, Open-Meteo boundary layer meteorology, and NASA MERRA-2 AOD extinction and surface PM₂.₅ reanalysis products. 
    *   **Key Operations:** Performs wind vector (U/V) decomposition, injects deterministic diurnal micro-physics features, and handles missing values to yield the final `master_hourly.csv` (21,937 records) and `master_daily_base.csv` (907 target-ready samples).

### Block 2: Machine Learning & Spatial Interfacing Pipeline
These scripts contain the core predictive algorithms, statistical rigor tests, and spatial bridging mechanics.

*   **`Week5_Model_Training.py`**
    *   **Purpose:** The primary machine learning module. Trains the XGBoost benchmark using a strict chronological `TimeSeriesSplit(n_splits=5)` to eliminate temporal look-ahead bias.
    *   **Diagnostics:** Computes SHAP (Shapley Additive Explanations) values for physical interpretability (generating the beeswarm summary plots). Runs the Diebold-Mariano and Ljung-Box statistical tests.
*   **`Week6_Spatial_Mapping_Collab_Version.py`**
    *   **Purpose:** The epidemiological interfacing module. Applies Inverse Distance Weighting (IDW) interpolation to disaggregate the singular city-wide PM₂.₅ forecast across 92 administrative Thana centroids. 
    *   **Outputs:** Calculates the Attributable Fraction (AF%) using non-linear Concentration-Response Functions (CRFs) and outputs the choropleth health risk maps via the `folium` library.

### Block 3: Retrieval-Augmented Generation (RAG) Policy Agent
These scripts manage the translation of numeric ML forecasts into legally grounded urban governance directives.

*   **`rag_run_full.py`** & **`RAG_RUN_FULL.ipynb`**
    *   **Purpose:** The canonical implementation of the GRAP-Dhaka Policy-Auditing RAG Agent.
    *   **Architecture:** Ingests the T+24h PM₂.₅ forecast and SHAP physical drivers. Retrieves relevant statutory context from a persistent `ChromaDB` vector store containing the Bangladesh *Air Pollution Control Rules (APCR) 2022*, *NAQMP 2024-2030*, and *WHO 2021 Guidelines* (embedded via `all-MiniLM-L6-v2`).
    *   **LLM Integration:** Orchestrates prompt synthesis via the Groq API using `openai/gpt-oss-120b` to generate hallucination-free Environmental Action Advisory Briefs (EAABs).
    *   **Evaluation:** Houses the local RAGAS-Proxy evaluation runner to compute semantic faithfulness, context precision, answer relevance, and hallucination rates.

---

## 🚀 Execution Note for Reviewers
While these scripts contain the complete modular codebase, **reviewers are highly encouraged to run the `MASTER_IMPLEMENTATION_DHAKA_PM25.ipynb` file located in the root repository directory.** The Master Colab Notebook orchestrates these components seamlessly into a single, pre-compiled "Execution Engine" for rapid reproducibility.

