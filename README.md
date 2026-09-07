# NexInsight — Your AI Data Analyst
### Datathon Problem Statement 3: The Automated Insight Analyst

NexInsight is an autonomous AI/ML platform that ingests unseen CSV and Excel spreadsheets, detects column types, performs data cleaning, calculates descriptive and multivariate statistics, identifies anomalies, generates dynamic Plotly visualizations, synthesizes factual AI insights, and provides an interactive "Ask Your Data" natural-language analyst.

---

## Key Features

1. **Zero-Assumption Architecture**: Never hardcodes column names, schemas, or domains. Adapts dynamically to any structured dataset.
2. **Automated Column Detection**: Accurately infers `Numeric`, `Categorical`, `Date`, `Boolean`, and `Text` types.
3. **Data Hygiene & Health Scoring**: Automatically eliminates duplicates, imputes missing values with median/mode, standardizes formatting, and computes a dynamic 0–100 Health Score.
4. **Statistical & ML Analysis**: Descriptive statistics, categorical dominance, Pearson correlation matrices, IQR/Z-score anomaly scoring, temporal trend analysis, and adaptive K-Means clustering.
5. **Interactive Visualizations**: Modern, responsive Plotly charts (time-series lines, aggregated bars, correlation heatmaps, scatter plots, donut distributions, histograms, anomaly inspection).
6. **Factual AI Insights**: Plain-English executive summaries and key findings generated strictly from computed results (zero hallucinations).
7. **Ask Your Data**: Conversational query engine that answers questions using live dataset calculations, supporting metrics, charts, and data slices.
8. **Professional SaaS Design**: Clean dark-neutral UI, soft shadows, rounded cards, custom Inter typography, and strictly zero emojis.

---

## Project Structure

```
NexInsight/
├── app.py                          # Main Streamlit SaaS application & router
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template (copy to .env)
├── create_test_datasets.py         # Script to regenerate test CSV/XLSX datasets
├── core/
│   ├── __init__.py
│   ├── data_processor.py           # Ingestion, auto-type detection, cleaning, health score
│   ├── analyzer.py                 # Descriptive stats, correlations, categorical, clustering
│   ├── visualizer.py               # Dynamic Plotly visualization engine
│   ├── insights.py                 # Factual AI insight synthesis engine
│   ├── anomalies.py                # IQR & Z-score anomaly detector with severity ranking
│   ├── qa_engine.py                # "Ask Your Data" natural language analytics engine
│   ├── rag_engine.py               # RAG pipeline for context-aware AI responses
│   ├── rag_retriever.py            # Retrieval layer for the RAG engine
│   ├── llm_client.py               # Unified LLM client (Groq API wrapper)
│   ├── grok_client.py              # Grok model client integration
│   └── batch_manager.py            # Multi-file batch upload & processing manager
├── static/
│   └── styles.css                  # Polished SaaS styling (cards, metrics, tables, dark-neutral)
├── test_datasets/
│   ├── dataset_a_sales.csv         # Numerical + Categorical + Date
│   ├── dataset_b_server_metrics.csv# Infrastructure server metrics
│   ├── dataset_c_survey_dirty.csv  # Missing values + duplicate rows
│   ├── dataset_d_categorical.csv   # Mostly categorical data
│   └── dataset_e_sensor_numeric.csv# Mostly numerical sensor telemetry
└── tests/
    ├── test_pipeline.py            # Core unit & integration tests across all 5 datasets
    ├── test_groq_rag.py            # Groq LLM + RAG engine integration tests
    ├── test_grok_rag.py            # Grok model RAG tests
    ├── test_batch_upload.py        # Batch upload & multi-file processing tests
    ├── test_olist_bench.py         # Benchmark tests on Olist e-commerce dataset
    ├── test_olist_qa.py            # QA engine tests on Olist dataset
    ├── test_html_cleanliness.py    # HTML output hygiene & XSS-safety tests
    └── test_editorial_ui.py        # UI editorial & rendering tests
```

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/atharva-923/NexInsight.git
   cd NexInsight
   ```

2. Copy the environment template and add your Groq API key:
   ```bash
   cp .env.example .env
   ```
   Then open `.env` and set:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
   Get a free key at [console.groq.com](https://console.groq.com).

---

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


2. Run the application:
   ```bash
   streamlit run app.py
   ```

3. Open your browser at:
   ```
   http://localhost:8501
   ```

---

## Running Automated Tests

Run individual test suites:

```bash
# Core pipeline — unit & integration tests on all 5 datasets
python -m pytest tests/test_pipeline.py

# Groq LLM + RAG engine integration
python -m pytest tests/test_groq_rag.py

# Batch upload & multi-file processing
python -m pytest tests/test_batch_upload.py

# QA engine on Olist e-commerce dataset
python -m pytest tests/test_olist_qa.py

# HTML output hygiene & XSS safety
python -m pytest tests/test_html_cleanliness.py
```

Or run the full suite at once:
```bash
python -m pytest tests/
```