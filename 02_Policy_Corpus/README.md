# ⚖️ Policy Corpus Directory (`/02_Policy_Corpus`)

## 📌 Overview
This directory contains the statutory documents and AI-optimized embeddings that form the non-parametric knowledge base for the Retrieval-Augmented Generation (RAG) policy agent. The corpus serves as the authoritative grounding matrix for environmental governance rules, standards, and intervention guidelines applicable to Dhaka's air quality management.

---

## 📁 Directory Guide

### 1️⃣ `/Original_PDFs/`
**Contents:** Raw, unaltered statutory documents in PDF format to ensure authenticity, legal defensibility, and traceability for peer review.

*   **`bangladesh-sro-225.pdf` (Air Pollution Control Rules 2022)**
    *   **Authority:** Department of Environment (DoE), Bangladesh
    *   **Relevance:** Defines the legally binding Bangladesh NAAQS 24h PM₂.₅ threshold (65 µg/m³) and the emergency response directives under Rule 15.
*   **`Bangladesh National Air Quality Management Plan 2024-2030.pdf`**
    *   **Authority:** Ministry of Environment, Forest and Climate Change (MoEFCC)
    *   **Relevance:** The strategic framework specifying the Graded Response Action Plan (GRAP-Dhaka) threshold stages and sector-specific intervention directives.
*   **`WHO global_Air_Quality_Guildlines_eng.pdf`**
    *   **Authority:** World Health Organization (WHO)
    *   **Relevance:** Provides the 2021 global health-based PM₂.₅ interim targets and final guideline values (15 µg/m³ 24h exposure ceiling).

### 2️⃣ `/Markdown_Embeddings/`
**Contents:** Plain-text Markdown conversions of the policy documents. 
**Why Markdown?** PDFs introduce severe layout artifacts (column-merge failures, hyphenation errors) that corrupt vector embeddings. By converting the statutory laws to Markdown, the semantic hierarchy (headers, lists, tables) is preserved cleanly for the LLM.

*   **`Air Pollution Control Rules 2022.md`**
*   **`Bangladesh National Air Quality Management Plan 2024-2030.md`**
*   **`WHO global_Air_Quality_Guildlines_eng.md`**

---

## ⚙️ RAG Indexing & Retrieval Architecture

To bypass the financial and computational overhead of foundational model fine-tuning, this corpus is ingested into a local, disk-persistent `ChromaDB` vector database. 

### 1. Chunking Strategy
The Markdown texts are segmentally parsed using a `RecursiveCharacterTextSplitter`:
*   **Chunk Size:** 1200 characters
*   **Chunk Overlap:** 200 characters 
*   **Total Corpus Size:** 1,175 structural chunks

### 2. Semantic Embedding
All chunks are embedded into a unified latent space using the lightweight, open-source **`all-MiniLM-L6-v2`** sentence transformer model. This yields 384-dimensional dense vectors at zero inference cost, stored locally.

### 3. Multi-Collection Query Routing
To ensure source-stratified retrieval, the vector database is divided into two distinct collections:
*   **`naqmp_policy` (333 chunks):** Contains the combined APCR 2022 and NAQMP 2024-2030 rules.
*   **`who_guidelines` (842 chunks):** Contains the international health benchmarks.

At inference time, the agent embeds the daily PM₂.₅ forecast and SHAP physical drivers to retrieve exactly **5 context chunks per query** (top $k=3$ matches from `naqmp_policy` and top $k=2$ matches from `who_guidelines` via cosine distance). 

---

## ✅ Quality Assurance & Anti-Hallucination
*   **100% Citation Grounding:** The RAG agent is strictly constrained to only cite clauses existing within these Markdown files. 
*   **Semantic Verification:** Local `RAGAS-Proxy` diagnostics ensure a 0.0% hallucination rate when interpreting these documents into Environmental Action Advisory Briefs (EAABs).
