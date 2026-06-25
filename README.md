# 🌍 Dhaka PM₂.₅ Forecasting & Policy-Auditing RAG Agent

## 📌 Project Overview
This repository contains the data, Python codebase, and execution environment for a multi-source air quality forecasting and environmental governance framework focused on the Dhaka metropolitan area, Bangladesh. 

The project bridges the gap between predictive machine learning, epidemiological health risk mapping, and actionable urban policy. It achieves this through a three-tier architecture:
1. **Multi-Source Data Fusion:** Integrating ground-level PM₂.₅ telemetry (OpenAQ), meteorological boundary conditions (Open-Meteo), and satellite reanalysis (NASA MERRA-2).
2. **Explainable ML & Geospatial Mapping:** Using XGBoost and SHAP to forecast and explain next-day (T+24h) PM₂.₅ concentrations, which are then spatially disaggregated to 92 administrative Thanas using Inverse Distance Weighting (IDW) to calculate public health risks (Attributable Fraction %).
3. **Retrieval-Augmented Generation (RAG) Policy Agent:** A specialized LLM agent grounded in the Bangladesh *Air Pollution Control Rules (APCR) 2022*, the *National Air Quality Management Plan (NAQMP) 2024-2030*, and *WHO 2021 Guidelines*. The agent automatically translates ML forecasts into hallucination-free Environmental Action Advisory Briefs (EAABs).

## 🚀 How to Run the Executable Portfolio
To ensure 100% computational reproducibility, prevent API-induced data drift, and bypass local hardware limits, the final implementation has been compiled into a single Google Colab Notebook.

1. Open the `MASTER_IMPLEMENTATION_DHAKA_PM25.ipynb` file located in the root directory.
2. Click **"Open in Colab"**.
3. Run the cells sequentially to reproduce the R² = 0.740 benchmark, the SHAP beeswarm plots, the geospatial maps, and the active RAG policy generation.

## ⚖️ Data and Code Availability Statement
The datasets and codebase provided here are available for academic peer review and reproducibility. Raw telemetry data was originally sourced from the OpenAQ platform, Open-Meteo Historical API, and the NASA Giovanni EarthData portal.

**GitHub Profile:** [https://github.com/Abid-Hossain29130](https://github.com/Abid-Hossain29130)

## 📁 Repository Structure

```
Dhaka-PM25-Forecasting-Policy-RAG-Agent/
├── 01_Data/
│   ├── Raw/
│   ├── Processed/
│   └── README.md
├── 02_Policy_Corpus/
│   ├── Original_PDFs/
│   ├── Markdown_Embeddings/
│   └── README.md
├── 03_Code/
│   └── README.md
├── 04_Results/
│   └── README.md
├── MASTER_IMPLEMENTATION_DHAKA_PM25.ipynb
└── README.md
```

## 📚 Quick Navigation
- **[01_Data/](./01_Data)** – Raw and processed datasets with provenance documentation
- **[02_Policy_Corpus/](./02_Policy_Corpus)** – Statutory documents and AI-optimized embeddings
- **[03_Code/](./03_Code)** – Standalone Python scripts for data engineering, modeling, and RAG
- **[04_Results/](./04_Results)** – Generated policy briefs and evaluation diagnostics

## 🔬 Technical Specifications
- **Model Architecture:** XGBoost with SHAP interpretability
- **Forecast Horizon:** T+24h next-day PM₂.₅ concentration (μg/m³)
- **Spatial Resolution:** 92 administrative Thanas in Dhaka
- **Vector DB:** ChromaDB for policy corpus indexing
- **LLM Framework:** RAG-based policy agent with hallucination mitigation
- **Computational Environment:** Google Colab (GPU-enabled)
- **Benchmark Performance:** R² = 0.740 on test set

## ✍️ Citation
If you use this repository for academic work, please cite:
```
@thesis{Hossain2026DhakaPM25,
  author = {Hossain, Abid},
  title = {From Prediction to Policy: An Explainable ML Framework for Health Risk Assessment and a RAG-LLM Agent for Urban Air Quality Management in Dhaka},
  school = {[National University of Bangladesh & Brit College of Engineering & Technology]},
  year = {2026}
}
```

## 📞 Contact & Contribution
For questions or collaboration opportunities, please reach out via GitHub Issues or directly at your contact information.

---
**Last Updated:** June 2026  
**Status:** Active Development  
**License:** [MIT License](LICENSE)
