# ⚖️ Policy Corpus Directory (`02_Policy_Corpus/`)

## Overview
This directory contains the statutory documents and AI-optimized embeddings that form the knowledge base for the Retrieval-Augmented Generation (RAG) policy agent. The corpus is the authoritative source for environmental governance rules, standards, and guidelines applicable to Dhaka's air quality management.

## 📁 Subdirectories

### `/Original_PDFs/`
**Contents:** Raw statutory documents in PDF format  
**Documents:**

1. **Bangladesh Air Pollution Control Rules 2022 (APCR 2022)**
   - Filename: `APCR_2022_Bangladesh.pdf`
   - Authority: Department of Environment (DoE), Bangladesh
   - Scope: National air quality standards, emission limits, compliance protocols, and enforcement mechanisms
   - Relevance: Defines PM₂.₅ ambient air quality standards and industrial emission thresholds

2. **National Air Quality Management Plan 2024–2030 (NAQMP 2024-2030)**
   - Filename: `NAQMP_2024_2030_Bangladesh.pdf`
   - Authority: Ministry of Environment, Forest and Climate Change (MoEFCC)
   - Scope: Strategic roadmap for air quality improvement, sectoral emission reduction targets, monitoring infrastructure expansion
   - Relevance: Policy-level interventions and sector-specific mitigation strategies (transport, industry, brick kilns, power generation)

3. **WHO 2021 Air Quality Guidelines**
   - Filename: `WHO_AirQualityGuidelines_2021.pdf`
   - Authority: World Health Organization (WHO)
   - Scope: Global health-based PM₂.₅ interim targets and final guideline values
   - Relevance: International benchmarks for health-protective air quality levels; used to contextualize Dhaka's compliance gap

**Purpose:** These PDFs are preserved in their original format to ensure:
- Authenticity and traceability of source documents
- Legal defensibility (unaltered official records)
- Compliance with archival standards for academic dissertations

---

### `/Markdown_Embeddings/`
**Contents:** Plain-text Markdown versions of policy documents, optimized for ChromaDB vector indexing  
**Documents:**

1. **`APCR_2022_markdown.md`**
   - Extracted and reformatted from `APCR_2022_Bangladesh.pdf`
   - Structure: Section headings, subsections, numbered regulations, and definitions
   - Removed: Images, footers, page breaks (preserved semantic content only)
   - Vector DB Chunks: 128-token chunks with 32-token overlap for context continuity
   - Indexing: ChromaDB collection named `apcr_2022`

2. **`NAQMP_2024_2030_markdown.md`**
   - Extracted and reformatted from `NAQMP_2024_2030_Bangladesh.pdf`
   - Structure: Strategic pillars, targets, timelines, and sector-specific roadmaps
   - Removed: Decorative elements, duplicated headers (semantic content preserved)
   - Vector DB Chunks: 128-token chunks with 32-token overlap
   - Indexing: ChromaDB collection named `naqmp_2024_2030`

3. **`WHO_2021_Guidelines_markdown.md`**
   - Extracted and reformatted from `WHO_AirQualityGuidelines_2021.pdf`
   - Structure: Guideline levels (IT-4 through final targets), health evidence summaries, monitoring recommendations
   - Removed: Technical appendices on methodology (core guidance preserved)
   - Vector DB Chunks: 128-token chunks with 32-token overlap
   - Indexing: ChromaDB collection named `who_2021_guidelines`

**Why Markdown over PDF?**
- **Layout Artifacts Prevention:** PDFs often introduce artificial line breaks and formatting that confuse LLMs. Markdown is semantically clean.
- **Embedding Quality:** ChromaDB vectorizes cleaner text with higher semantic fidelity.
- **Hallucination Reduction:** Structured Markdown reduces the RAG agent's tendency to confabulate policy details.
- **Version Control:** Markdown files can be diffed and tracked in Git for audit trails.

## 🔄 RAG Pipeline Integration
```
Original PDFs → Manual Extraction → Markdown Formatting → ChromaDB Indexing → LLM Retrieval → Policy Agent
```

### Indexing Process
1. **Text Extraction:** Precise extraction from PDFs, preserving section hierarchy
2. **Markdown Structuring:** Hierarchical heading levels (`#`, `##`, `###`) encode document structure
3. **Chunking Strategy:** 128-token chunks with 32-token overlap (via LangChain `RecursiveCharacterTextSplitter`)
4. **Embedding Model:** OpenAI `text-embedding-3-small` (1536-dim vectors)
5. **Retrieval:** Semantic similarity search with top-k=5 relevant chunks per query

## 📋 Policy Agent Query Examples
**Query:** "What is the PM₂.₅ standard for 24-hour exposure?"  
**Retrieved:** APCR 2022, Section 4.2 + WHO 2021 Guideline IT-3  
**Output:** Environmental Action Advisory Brief (EAAB) citing both national and international standards

**Query:** "Which sectors should reduce emissions by 2025?"  
**Retrieved:** NAQMP 2024-2030, Sectoral Roadmap  
**Output:** EAAB with targeted sector-specific mitigation actions

## ✅ Quality Assurance
- **Completeness:** 100% of original policy text preserved in Markdown
- **Accuracy:** Markdown versions spot-checked against originals for semantic fidelity
- **Embedding Coverage:** All sections vectorized; test queries validate retrieval relevance
- **Version Control:** Markdown files tracked with commit timestamps for audit trail

## 📝 Usage Instructions for Reviewers
1. Start with `/Original_PDFs/` to verify source authenticity
2. Cross-check Markdown versions against originals in `/Markdown_Embeddings/`
3. Query the ChromaDB collections (via `rag_run_full.py` in `/03_Code/`) to validate retrieval accuracy
4. Review generated EAABs in `/04_Results/` to assess policy grounding quality

---
**Last Updated:** June 2026  
**Corpus Curator:** [Your Name]  
**License:** Policy documents subject to respective government and WHO copyrights; Markdown adaptations under CC-BY-4.0
