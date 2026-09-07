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
├── app.py                     # Main Streamlit SaaS application & router
├── requirements.txt           # Python dependencies
├── core/
│   ├── __init__.py
│   ├── data_processor.py      # Ingestion, auto-type detection, cleaning, health score
│   ├── analyzer.py            # Descriptive stats, correlations, categorical, clustering
│   ├── visualizer.py          # Dynamic Plotly visualization engine
│   ├── insights.py            # Factual AI insight synthesis engine
│   ├── anomalies.py           # IQR & Z-score anomaly detector with severity ranking
│   └── qa_engine.py           # "Ask Your Data" natural language analytics engine
├── static/
│   └── styles.css             # Polished SaaS styling (cards, metrics, tables, dark-neutral)
├── test_datasets/
│   ├── dataset_a_sales.csv            # Numerical + Categorical + Date
│   ├── dataset_b_server_metrics.csv   # Infrastructure server metrics
│   ├── dataset_c_survey_dirty.csv     # Missing values + duplicate rows
│   ├── dataset_d_categorical.csv      # Mostly categorical data
│   └── dataset_e_sensor_numeric.csv   # Mostly numerical sensor telemetry
└── tests/
    └── test_pipeline.py       # Automated unit & integration tests on all 5 datasets
```

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

Run the test suite across all 5 test datasets:
```bash
python tests/test_pipeline.py
```