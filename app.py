"""
NexInsight — Your AI Data Analyst
Problem Statement 3: The Automated Insight Analyst
Autonomous end-to-end dataset understanding, cleaning, analysis, visualization, and conversational intelligence.
Redesigned with Modern Premium Fintech / Enterprise SaaS Aesthetic.
"""

import os
import io
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="NexInsight — AI Data Analyst",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load core modules
from core.data_processor import DataProcessor
from core.analyzer import DataAnalyzer
from core.visualizer import Visualizer
from core.insights import InsightEngine
from core.anomalies import AnomalyDetector
from core.qa_engine import DataQAEngine

# Load custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "static", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Session State Initialization
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "column_types" not in st.session_state:
    st.session_state.column_types = {}
if "cleaning_summary" not in st.session_state:
    st.session_state.cleaning_summary = {}
if "health_metrics" not in st.session_state:
    st.session_state.health_metrics = {}
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "ai_insights" not in st.session_state:
    st.session_state.ai_insights = {}
if "anomalies_data" not in st.session_state:
    st.session_state.anomalies_data = {}
if "llm_api_key" not in st.session_state:
    st.session_state.llm_api_key = ""
if "remove_duplicates" not in st.session_state:
    st.session_state.remove_duplicates = True


def process_dataset(file_obj, filename: str):
    """Orchestrates the entire automated ingestion, cleaning, analysis, and insight generation."""
    processor = DataProcessor()
    proc_res = processor.process(file_obj, filename, remove_duplicates=st.session_state.remove_duplicates)

    cleaned_df = proc_res["cleaned_df"]
    raw_df = proc_res["raw_df"]
    col_types = proc_res["column_types"]
    clean_summary = proc_res["cleaning_summary"]
    health = proc_res["health_metrics"]

    # Run Analysis
    analyzer = DataAnalyzer(cleaned_df, col_types)
    analysis_results = analyzer.run_full_analysis()

    # Run Dedicated Anomaly Detection
    detector = AnomalyDetector(cleaned_df, col_types)
    anomalies_data = detector.get_comprehensive_anomalies()

    # Generate Factual AI Insights
    insights = InsightEngine.generate_factual_insights(
        analysis_results, health, clean_summary
    )

    # Store in session state
    st.session_state.current_file_name = filename
    st.session_state.raw_df = raw_df
    st.session_state.cleaned_df = cleaned_df
    st.session_state.column_types = col_types
    st.session_state.cleaning_summary = clean_summary
    st.session_state.health_metrics = health
    st.session_state.analysis_results = analysis_results
    st.session_state.anomalies_data = anomalies_data
    st.session_state.ai_insights = insights


# Automatically load default Dataset A if nothing loaded yet
if st.session_state.cleaned_df is None:
    default_path = os.path.join("test_datasets", "dataset_a_sales.csv")
    if os.path.exists(default_path):
        process_dataset(default_path, "dataset_a_sales.csv")


# SIDEBAR
st.sidebar.markdown(
    """
    <div class="sidebar-brand-box">
        <div class="brand-title-wrap">
            <h2 class="brand-title" style="font-size: 1.35rem;">NexInsight</h2>
            <span class="brand-badge" style="background: #2563EB;">AI ANALYST</span>
        </div>
        <p style="font-size: 0.76rem; color: #6B7280; margin: 0.35rem 0 0 0; font-weight: 500;">Autonomous Data Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<div class='sidebar-nav-title'>Navigation</div>", unsafe_allow_html=True)

nav_page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Data Explorer", "Insights", "Anomalies", "AI Analyst", "Settings"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-nav-title'>Test Datasets</div>", unsafe_allow_html=True)

sample_options = {
    "Dataset A (Sales & Trends)": "test_datasets/dataset_a_sales.csv",
    "Dataset B (Server Metrics)": "test_datasets/dataset_b_server_metrics.csv",
    "Dataset C (Survey - Dirty)": "test_datasets/dataset_c_survey_dirty.csv",
    "Dataset D (Categorical)": "test_datasets/dataset_d_categorical.csv",
    "Dataset E (Sensors - Numeric)": "test_datasets/dataset_e_sensor_numeric.csv"
}

selected_sample = st.sidebar.selectbox(
    "Quick Switch Dataset",
    list(sample_options.keys()),
    index=0,
    label_visibility="collapsed"
)

if st.sidebar.button("Load Dataset", use_container_width=True):
    target_path = sample_options[selected_sample]
    if os.path.exists(target_path):
        filename = os.path.basename(target_path)
        with st.spinner("Processing dataset..."):
            process_dataset(target_path, filename)
        st.rerun()

# Sidebar Active Dataset Meta Box
if st.session_state.current_file_name:
    h = st.session_state.health_metrics
    clean_df = st.session_state.cleaned_df
    num_c = len([c for c, t in st.session_state.column_types.items() if t == 'Numeric'])
    cat_c = len([c for c, t in st.session_state.column_types.items() if t in ['Categorical', 'Boolean']])
    st.sidebar.markdown(
        f"""
        <div class="sidebar-dataset-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.7rem; text-transform: uppercase; color: #6B7280; font-weight: 700; letter-spacing: 0.05em;">Active Dataset</span>
                <span class="pill-badge green">
                    <span class="live-dot" style="width: 5px; height: 5px;"></span> {h.get('score', 0)}/100
                </span>
            </div>
            <div style="font-size: 0.85rem; font-weight: 700; color: #111111; margin-top: 6px; word-break: break-all;">
                {st.session_state.current_file_name}
            </div>
            <div style="font-size: 0.74rem; color: #6B7280; margin-top: 4px;">
                {len(clean_df):,} rows · {len(clean_df.columns)} attributes<br>
                <span style="color: #9CA3AF;">{num_c} numeric · {cat_c} categorical</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# TOP APP BAR (Zentra inspiration: large heading + meta pills)
st.markdown(
    f"""
    <div class="overview-heading-row">
        <div>
            <h1 class="overview-heading">{nav_page}</h1>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div class="dataset-meta-pill">
                <span class="live-dot"></span>
                <span>Active: <strong style="color: #111111;">{st.session_state.current_file_name or "None"}</strong></span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# PAGE 1: OVERVIEW (MAIN DASHBOARD)
# ==============================================================================
if nav_page == "Overview":
    if st.session_state.cleaned_df is None:
        st.warning("Please upload or select a dataset to begin.")
        st.stop()

    df = st.session_state.cleaned_df
    raw_df = st.session_state.raw_df
    h = st.session_state.health_metrics
    clean_sum = st.session_state.cleaning_summary
    anoms = st.session_state.anomalies_data
    insights = st.session_state.ai_insights
    col_types = st.session_state.column_types

    num_cols = [c for c, t in col_types.items() if t == 'Numeric']
    cat_cols = [c for c, t in col_types.items() if t in ['Categorical', 'Boolean']]
    date_cols = [c for c, t in col_types.items() if t == 'Date']

    # 1. Top 5 Metric Cards (as requested: Data Health, Records, Attributes, Missing, Anomalies)
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card-fintech">
                <div class="metric-label-compact">
                    <span>Data Health</span>
                    <span class="pill-badge green">{h.get('status', 'Excellent')}</span>
                </div>
                <div class="metric-number-lg">{h.get('score', 100)}<span style="font-size: 1rem; color: #9CA3AF; font-weight: 500;">/100</span></div>
                <div class="metric-subtext-clean">Overall quality index</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card-fintech">
                <div class="metric-label-compact">
                    <span>Total Records</span>
                    <span class="pill-badge blue">Cleaned</span>
                </div>
                <div class="metric-number-lg">{len(df):,}</div>
                <div class="metric-subtext-clean">{len(raw_df):,} raw entries</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card-fintech">
                <div class="metric-label-compact">
                    <span>Attributes</span>
                    <span class="pill-badge blue">{len(df.columns)}</span>
                </div>
                <div class="metric-number-lg">{len(df.columns)}</div>
                <div class="metric-subtext-clean">{len(num_cols)} num · {len(cat_cols)} cat · {len(date_cols)} date</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card-fintech">
                <div class="metric-label-compact">
                    <span>Missing Values</span>
                    <span class="pill-badge amber">{clean_sum.get('missing_values_handled', 0)}</span>
                </div>
                <div class="metric-number-lg">{h.get('total_missing', 0):,}</div>
                <div class="metric-subtext-clean">{clean_sum.get('missing_values_handled', 0):,} resolved & imputed</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c5:
        tot_anom = anoms.get('total_anomalies', 0)
        crit_anom = anoms.get('severity_counts', {}).get('Critical', 0)
        mod_anom = anoms.get('severity_counts', {}).get('Moderate', 0)
        badge_cls = "red" if crit_anom > 0 else "blue"
        st.markdown(
            f"""
            <div class="metric-card-fintech">
                <div class="metric-label-compact">
                    <span>Anomalies</span>
                    <span class="pill-badge {badge_cls}">{crit_anom} Critical</span>
                </div>
                <div class="metric-number-lg">{tot_anom:,}</div>
                <div class="metric-subtext-clean">{crit_anom} critical · {mod_anom} moderate</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 2. Middle Row: Primary Trend/Analysis (Large Left) + Supporting Distribution (Right)
    auto_charts = Visualizer.auto_generate_dashboard_charts(df, col_types)

    chart_col_left, chart_col_right = st.columns([1.75, 1.25])

    with chart_col_left:
        st.markdown(
            """
            <div class="white-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <div>
                        <h3 class="card-title-fintech">Primary Trend / Analysis</h3>
                        <div class="card-subtitle-fintech">Dominant temporal trajectory or numerical interaction</div>
                    </div>
                    <span class="pill-badge blue">PRIMARY</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        if auto_charts and len(auto_charts) > 0:
            st.plotly_chart(auto_charts[0], use_container_width=True)
        else:
            st.info("No numerical features available for trend analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col_right:
        st.markdown(
            """
            <div class="white-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <div>
                        <h3 class="card-title-fintech">Distribution</h3>
                        <div class="card-subtitle-fintech">Categorical share or feature dispersion</div>
                    </div>
                    <span class="pill-badge blue">SEGMENTATION</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        if auto_charts and len(auto_charts) > 1:
            st.plotly_chart(auto_charts[1], use_container_width=True)
        elif auto_charts and len(auto_charts) > 0:
            st.plotly_chart(auto_charts[0], use_container_width=True)
        else:
            st.info("No categorical or distribution features available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Bottom Row: 3 White Cards (AI INSIGHTS | DATA QUALITY | ANOMALIES)
    b1, b2, b3 = st.columns(3)

    with b1:
        st.markdown(
            f"""
            <div class="white-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <h3 class="card-title-fintech">AI Insights</h3>
                    <span class="pill-badge blue">SYNTHESIS</span>
                </div>
                <div class="ai-summary-box">
                    "{insights.get('executive_summary', 'Dataset evaluated with clean metrics.')}"
                </div>
                <div class="insight-sub-card">
                    <div class="insight-sub-tag finding">KEY FINDING</div>
                    <div class="insight-sub-text">{insights.get('key_findings', ['Attributes evaluated cleanly.'])[0]}</div>
                </div>
                <div class="insight-sub-card">
                    <div class="insight-sub-tag trend">TREND / PATTERN</div>
                    <div class="insight-sub-text">{insights.get('important_patterns', ['Distribution is balanced across records.'])[0]}</div>
                </div>
                <div class="insight-sub-card">
                    <div class="insight-sub-tag risk">RISK / OBSERVATION</div>
                    <div class="insight-sub-text">{insights.get('anomalies', ['No severe distribution anomalies noted.'])[0]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b2:
        comp_pct = h.get('completeness_pct', 100.0)
        uniq_pct = h.get('uniqueness_pct', 100.0)
        val_pct = h.get('validity_pct', 100.0)
        st.markdown(
            f"""
            <div class="white-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <h3 class="card-title-fintech">Data Quality</h3>
                    <span class="pill-badge green">{h.get('score', 100)}/100</span>
                </div>
                <div class="quality-row">
                    <span style="color: #6B7280; font-weight: 500;">Completeness</span>
                    <div>
                        <strong style="color: #111111;">{comp_pct}%</strong>
                        <div class="quality-bar-wrap"><div class="quality-bar-fill" style="width: {comp_pct}%;"></div></div>
                    </div>
                </div>
                <div class="quality-row">
                    <span style="color: #6B7280; font-weight: 500;">Uniqueness</span>
                    <div>
                        <strong style="color: #111111;">{uniq_pct}%</strong>
                        <div class="quality-bar-wrap"><div class="quality-bar-fill" style="width: {uniq_pct}%;"></div></div>
                    </div>
                </div>
                <div class="quality-row">
                    <span style="color: #6B7280; font-weight: 500;">Structure Validity</span>
                    <div>
                        <strong style="color: #111111;">{val_pct}%</strong>
                        <div class="quality-bar-wrap"><div class="quality-bar-fill" style="width: {val_pct}%;"></div></div>
                    </div>
                </div>
                <div class="quality-row">
                    <span style="color: #6B7280; font-weight: 500;">Duplicates Dropped</span>
                    <strong style="color: #111111;">{clean_sum.get('duplicates_removed', 0):,}</strong>
                </div>
                <div class="quality-row">
                    <span style="color: #6B7280; font-weight: 500;">Missing Imputed</span>
                    <strong style="color: #111111;">{clean_sum.get('missing_values_handled', 0):,}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b3:
        anom_list = anoms.get("anomalies_list", [])
        top_anom = anom_list[0] if anom_list else None
        top_col = top_anom["column"] if top_anom else "None"
        top_val = f"{top_anom['value']:,}" if top_anom else "0"
        top_reason = top_anom["reason"] if top_anom else "All parameters within normal standard deviation."
        
        st.markdown(
            f"""
            <div class="white-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <h3 class="card-title-fintech">Anomalies</h3>
                    <span class="pill-badge {'red' if crit_anom > 0 else 'blue'}">{tot_anom} Detected</span>
                </div>
                <div style="display: flex; gap: 8px; margin-bottom: 0.85rem;">
                    <div style="flex: 1; background: #F8FAFC; border: 1px solid #E5E7EB; border-radius: 8px; padding: 6px 10px; text-align: center;">
                        <span style="font-size: 0.68rem; color: #6B7280; font-weight: 700; text-transform: uppercase;">Critical</span>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #EF4444;">{crit_anom}</div>
                    </div>
                    <div style="flex: 1; background: #F8FAFC; border: 1px solid #E5E7EB; border-radius: 8px; padding: 6px 10px; text-align: center;">
                        <span style="font-size: 0.68rem; color: #6B7280; font-weight: 700; text-transform: uppercase;">Moderate</span>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #F59E0B;">{mod_anom}</div>
                    </div>
                    <div style="flex: 1; background: #F8FAFC; border: 1px solid #E5E7EB; border-radius: 8px; padding: 6px 10px; text-align: center;">
                        <span style="font-size: 0.68rem; color: #6B7280; font-weight: 700; text-transform: uppercase;">Low</span>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #3B82F6;">{anoms.get('severity_counts', {}).get('Low', 0)}</div>
                    </div>
                </div>
                <div class="insight-sub-card" style="border-left: 3px solid {'#EF4444' if crit_anom > 0 else '#3B82F6'};">
                    <div class="insight-sub-tag risk">MOST SEVERE ANOMALY</div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #111111; margin-top: 2px;">{top_col}: {top_val}</div>
                    <div class="insight-sub-text" style="font-size: 0.76rem; color: #6B7280; margin-top: 2px;">{top_reason}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 4. Ask Your Data Prompt Bar (Inspired by the Reference Image Prompt Input)
    st.markdown(
        """
        <div class="ask-data-container">
            <div class="ask-data-prompt-label">
                <span style="color: #2563EB; font-weight: 700;">Ask NexInsight:</span>
                <span>What would you like to explore next?</span>
            </div>
        """,
        unsafe_allow_html=True
    )

    dash_q = st.text_input(
        "Ask Your Data",
        placeholder="e.g. What is the highest-performing category? What caused unusual spikes? What are the strongest correlations?",
        label_visibility="collapsed",
        key="dash_prompt_box"
    )

    if dash_q:
        qa_engine = DataQAEngine(df, col_types, st.session_state.analysis_results, h)
        with st.spinner("Analyzing live dataset parameters..."):
            ans_data = qa_engine.answer_query(dash_q)

        st.markdown(
            f"""
            <div class="ask-data-response">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <span style="font-size: 0.72rem; text-transform: uppercase; color: #2563EB; font-weight: 700; letter-spacing: 0.06em;">NexInsight Analyst Response</span>
                    {f'<span class="pill-badge blue">{ans_data["metric_highlight"]}</span>' if ans_data.get("metric_highlight") else ''}
                </div>
                <div style="font-size: 0.9rem; line-height: 1.6; color: #111111;">{ans_data['answer']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if ans_data.get("figure"):
            st.plotly_chart(ans_data["figure"], use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: DATA EXPLORER (DATA UPLOAD & SCHEMA)
# ==============================================================================
elif nav_page == "Data Explorer":
    st.markdown(
        """
        <div class="white-card" style="margin-bottom: 1.2rem;">
            <h3 class="card-title-fintech" style="font-size: 1.15rem;">Upload & Ingestion Pipeline</h3>
            <div class="card-subtitle-fintech">Drop any unseen CSV or XLSX file. NexInsight automatically parses schema, detects anomalies, and cleans hygiene.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    if uploaded_file is not None:
        if st.button("Process & Analyze Uploaded Dataset", type="primary", use_container_width=True):
            with st.spinner("Ingesting and analyzing dataset..."):
                process_dataset(uploaded_file, uploaded_file.name)
            st.success(f"Successfully processed {uploaded_file.name}")
            st.rerun()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df
        raw_df = st.session_state.raw_df
        clean_sum = st.session_state.cleaning_summary
        h = st.session_state.health_metrics

        # Health & Hygiene Summary Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"<div class='metric-card-fintech'><div class='metric-label-compact'><span>Health Score</span><span class='pill-badge green'>{h.get('status', 'Excellent')}</span></div><div class='metric-number-lg'>{h.get('score', 100)}/100</div><div class='metric-subtext-clean'>Quality score index</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='metric-card-fintech'><div class='metric-label-compact'><span>Duplicates Dropped</span></div><div class='metric-number-lg'>{clean_sum.get('duplicates_removed', 0):,}</div><div class='metric-subtext-clean'>Out of {h.get('total_rows', len(df)):,} rows</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='metric-card-fintech'><div class='metric-label-compact'><span>Missing Imputed</span></div><div class='metric-number-lg'>{clean_sum.get('missing_values_handled', 0):,}</div><div class='metric-subtext-clean'>Median & mode imputation</div></div>", unsafe_allow_html=True)
        with k4:
            st.markdown(f"<div class='metric-card-fintech'><div class='metric-label-compact'><span>Dates Detected</span></div><div class='metric-number-lg'>{clean_sum.get('date_columns_detected', 0)}</div><div class='metric-subtext-clean'>Standardized temporal series</div></div>", unsafe_allow_html=True)

        # Cleaning Audit Trail Accordion
        if clean_sum.get("details"):
            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            with st.expander("Data Hygiene Audit Summary", expanded=True):
                for d in clean_sum["details"]:
                    st.markdown(f"- {d}")

        # Detected Column Schema Table
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 class='card-title-fintech' style='margin-bottom: 0.6rem;'>Inferred Schema & Attribute Typings</h3>", unsafe_allow_html=True)
        
        schema_rows = []
        for col, ctype in st.session_state.column_types.items():
            sample_val = str(df[col].iloc[0]) if len(df) > 0 else "N/A"
            unq_cnt = df[col].nunique()
            null_cnt = raw_df[col].isna().sum() if col in raw_df.columns else 0
            schema_rows.append({
                "Column Name": col,
                "Detected Type": ctype,
                "Unique Values": f"{unq_cnt:,}",
                "Raw Missing": f"{null_cnt:,}",
                "Sample Value": sample_val[:45]
            })

        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

        # Raw vs Cleaned Preview Toggle
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        view_opt = st.radio("Inspect Data Sample:", ["Cleaned Data", "Raw Data"], horizontal=True)
        if view_opt == "Cleaned Data":
            st.dataframe(df.head(100), use_container_width=True)
        else:
            st.dataframe(raw_df.head(100), use_container_width=True)


# ==============================================================================
# PAGE 3: INSIGHTS
# ==============================================================================
elif nav_page == "Insights":
    if st.session_state.cleaned_df is None:
        st.warning("Please upload or load a dataset first.")
        st.stop()

    df = st.session_state.cleaned_df
    insights = st.session_state.ai_insights
    analysis = st.session_state.analysis_results

    st.markdown(
        """
        <div class="white-card" style="margin-bottom: 1.2rem;">
            <h3 class="card-title-fintech" style="font-size: 1.15rem;">Strategic & AI-Generated Insights</h3>
            <div class="card-subtitle-fintech">Rigorous statistical synthesis calculated strictly from live data parameters (zero hallucinations).</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Executive Brief Card
    st.markdown(
        f"""
        <div class="white-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                <span class="metric-label-compact" style="margin: 0;">Executive Overview</span>
                <span class="pill-badge green">VERIFIED FACTS</span>
            </div>
            <div class="ai-summary-box" style="font-size: 0.95rem;">
                {insights.get('executive_summary', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3-Column Structured Breakdown
    c_find, c_pat, c_obs = st.columns(3)

    with c_find:
        st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #111111; margin-bottom: 0.6rem;'>Key Statistical Findings</div>", unsafe_allow_html=True)
        for item in insights.get("key_findings", []):
            st.markdown(f"<div class='insight-sub-card'><div class='insight-sub-tag finding'>FINDING</div><div class='insight-sub-text'>{item}</div></div>", unsafe_allow_html=True)

    with c_pat:
        st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #111111; margin-bottom: 0.6rem;'>Dominant Patterns</div>", unsafe_allow_html=True)
        for item in insights.get("important_patterns", []):
            st.markdown(f"<div class='insight-sub-card'><div class='insight-sub-tag trend'>PATTERN</div><div class='insight-sub-text'>{item}</div></div>", unsafe_allow_html=True)

    with c_obs:
        st.markdown("<div style='font-size: 0.9rem; font-weight: 700; color: #111111; margin-bottom: 0.6rem;'>Actionable Observations</div>", unsafe_allow_html=True)
        for item in insights.get("observations", []):
            st.markdown(f"<div class='insight-sub-card'><div class='insight-sub-tag risk'>ACTION</div><div class='insight-sub-text'>{item}</div></div>", unsafe_allow_html=True)

    # Correlation Matrix Section
    corr_data = analysis.get("correlations", {})
    if corr_data.get("has_correlation"):
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 class='card-title-fintech' style='margin-bottom: 0.6rem;'>Cross-Feature Correlations & Dependencies</h3>", unsafe_allow_html=True)
        c_heat, c_pairs = st.columns([1.3, 1])

        with c_heat:
            st.plotly_chart(Visualizer.create_correlation_heatmap(corr_data["matrix_df"]), use_container_width=True)

        with c_pairs:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #111111; margin-bottom: 0.5rem;'>Strongest Interacting Pairs</div>", unsafe_allow_html=True)
            pairs = corr_data.get("significant_pairs", [])
            if pairs:
                p_df = pd.DataFrame([
                    {"Feature 1": p["col1"], "Feature 2": p["col2"], "Correlation": f"{p['correlation']:.2f}", "Direction": f"{p['strength']} {p['direction']}"}
                    for p in pairs[:8]
                ])
                st.dataframe(p_df, use_container_width=True, hide_index=True)
            else:
                st.info("No significant cross-correlation detected between numerical attributes.")

    # Clustering Section
    clustering = analysis.get("clustering")
    if clustering and clustering.get("applicable"):
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="white-card" style="margin-bottom: 0.8rem;">
                <h3 class="card-title-fintech">Adaptive K-Means Behavioral Clustering</h3>
                <div class="card-subtitle-fintech">Segmented records into {clustering['n_clusters']} cohesive natural clusters (Silhouette Score: {clustering['silhouette_score']})</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        cl_chart, cl_prof = st.columns([1.3, 1])
        with cl_chart:
            st.plotly_chart(Visualizer.create_cluster_scatter(clustering["pca_x"], clustering["pca_y"], clustering["labels"]), use_container_width=True)

        with cl_prof:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #111111; margin-bottom: 0.5rem;'>Segment Characteristics</div>", unsafe_allow_html=True)
            for p in clustering["profiles"]:
                st.markdown(
                    f"""
                    <div class="insight-sub-card">
                        <div style="font-weight: 700; color: #2563EB;">{p['name']} <span style="font-size: 0.76rem; color: #6B7280; font-weight: 400;">({p['percentage']}% of records)</span></div>
                        <div style="font-size: 0.78rem; color: #374151; margin-top: 3px;">Traits: {p['key_traits']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ==============================================================================
# PAGE 4: ANOMALIES
# ==============================================================================
elif nav_page == "Anomalies":
    if st.session_state.cleaned_df is None:
        st.warning("Please upload or load a dataset first.")
        st.stop()

    df = st.session_state.cleaned_df
    anoms = st.session_state.anomalies_data

    st.markdown(
        """
        <div class="white-card" style="margin-bottom: 1.2rem;">
            <h3 class="card-title-fintech" style="font-size: 1.15rem;">Anomaly & Statistical Outlier Intelligence</h3>
            <div class="card-subtitle-fintech">Dual-method outlier detection leveraging 1.5x Interquartile Range (IQR) and Z-score (>3.0) distribution boundaries.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Anomaly KPI Cards with Severity Indicators
    a1, a2, a3, a4 = st.columns(4)
    tot = anoms.get("total_anomalies", 0)
    crit = anoms.get("severity_counts", {}).get("Critical", 0)
    mod = anoms.get("severity_counts", {}).get("Moderate", 0)
    low = anoms.get("severity_counts", {}).get("Low", 0)

    with a1:
        st.markdown(f"<div class='metric-card-fintech'><div class='metric-label-compact'><span>Total Flagged</span></div><div class='metric-number-lg'>{tot:,}</div><div class='metric-subtext-clean'>Across numerical features</div></div>", unsafe_allow_html=True)
    with a2:
        st.markdown(f"<div class='metric-card-fintech'><div class='metric-label-compact'><span>Critical Severity</span><span class='pill-badge red'>Z > 3.5</span></div><div class='metric-number-lg' style='color: #EF4444;'>{crit}</div><div class='metric-subtext-clean'>Requires immediate review</div></div>", unsafe_allow_html=True)
    with a3:
        st.markdown(f"<div class='metric-card-fintech'><div class='metric-label-compact'><span>Moderate Severity</span><span class='pill-badge amber'>1.5x IQR</span></div><div class='metric-number-lg' style='color: #F59E0B;'>{mod}</div><div class='metric-subtext-clean'>Outside standard range</div></div>", unsafe_allow_html=True)
    with a4:
        st.markdown(f"<div class='metric-card-fintech'><div class='metric-label-compact'><span>Low / Edge</span><span class='pill-badge blue'>Boundary</span></div><div class='metric-number-lg' style='color: #3B82F6;'>{low}</div><div class='metric-subtext-clean'>Distribution boundaries</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    anom_list = anoms.get("anomalies_list", [])
    if anom_list:
        # Visual Anomaly Scatter Inspector
        num_cols_with_anoms = list(set([a["column"] for a in anom_list]))
        selected_anom_col = st.selectbox("Inspect Attribute Outliers:", num_cols_with_anoms)
        if selected_anom_col:
            st.plotly_chart(Visualizer.create_anomaly_scatter(df, selected_anom_col, anom_list), use_container_width=True)

        # Highlight Top 3 Most Extreme Anomalies in Cards
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 class='card-title-fintech' style='margin-bottom: 0.6rem;'>Most Severe Outlier Records</h3>", unsafe_allow_html=True)
        top_cards_cols = st.columns(min(3, len(anom_list)))
        for i, c in enumerate(top_cards_cols):
            rec = anom_list[i]
            badge_color = "red" if rec["severity"] == "Critical" else "amber"
            with c:
                st.markdown(
                    f"""
                    <div class="white-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 0.74rem; text-transform: uppercase; color: #6B7280; font-weight: 700;">{rec['column']}</span>
                            <span class="pill-badge {badge_color}">{rec['severity']}</span>
                        </div>
                        <div style="font-size: 1.65rem; font-weight: 800; color: #111111; margin: 0.4rem 0;">{rec['value']:,}</div>
                        <div style="font-size: 0.76rem; color: #6B7280;">Normal: <strong style="color: #15803D;">{rec['normal_range']}</strong></div>
                        <div style="font-size: 0.74rem; color: #6B7280; margin-top: 0.4rem; line-height: 1.35;">{rec['reason']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Full Anomaly Records Table
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 class='card-title-fintech' style='margin-bottom: 0.6rem;'>Complete Anomaly Register</h3>", unsafe_allow_html=True)
        table_df = pd.DataFrame(anom_list)[["row_index", "column", "value", "normal_range", "z_score", "severity", "reason"]]
        table_df.columns = ["Row #", "Column", "Value", "Normal Range", "Z-Score", "Severity", "Explanation"]
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    else:
        st.success("No anomalies detected in the current numerical parameters.")


# ==============================================================================
# PAGE 5: AI ANALYST ("ASK YOUR DATA")
# ==============================================================================
elif nav_page == "AI Analyst":
    if st.session_state.cleaned_df is None:
        st.warning("Please upload or load a dataset first.")
        st.stop()

    df = st.session_state.cleaned_df

    st.markdown(
        """
        <div class="white-card" style="margin-bottom: 1.2rem;">
            <h3 class="card-title-fintech" style="font-size: 1.15rem;">AI Analyst — Ask Your Data</h3>
            <div class="card-subtitle-fintech">Query dataset attributes in plain English. Answers are evaluated dynamically against live calculations without static scripts.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Preset Quick Inquiry Chips
    st.markdown("<p style='font-size: 0.74rem; font-weight: 700; text-transform: uppercase; color: #6B7280; letter-spacing: 0.06em; margin-bottom: 8px;'>Suggested Inquiries:</p>", unsafe_allow_html=True)
    
    col_chips = st.columns(5)
    sample_queries = [
        "What is the highest-performing category?",
        "Which month had the highest value?",
        "Are there unusual records?",
        "What are the strongest correlations?",
        "What are the most important patterns in this dataset?"
    ]

    selected_chip_query = None
    for idx, (c, q_text) in enumerate(zip(col_chips, sample_queries)):
        with c:
            if st.button(q_text, key=f"chip_{idx}", use_container_width=True):
                selected_chip_query = q_text

    query_input = st.text_input(
        "Type your analytical question:",
        value=selected_chip_query or "",
        placeholder="e.g. What is the average value? Which category dominates?",
        key="qa_input_box"
    )

    if query_input:
        qa_engine = DataQAEngine(
            df,
            st.session_state.column_types,
            st.session_state.analysis_results,
            st.session_state.health_metrics
        )
        with st.spinner("Analyzing live dataset parameters..."):
            ans_res = qa_engine.answer_query(query_input)

        st.markdown(
            f"""
            <div class="white-card" style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.74rem; text-transform: uppercase; color: #2563EB; font-weight: 700; letter-spacing: 0.06em;">NexInsight Intelligence Output</span>
                    {f'<span class="pill-badge blue">{ans_res["metric_highlight"]}</span>' if ans_res.get("metric_highlight") else ''}
                </div>
                <div style="font-size: 0.92rem; line-height: 1.6; color: #111111;">{ans_data if 'ans_data' in locals() and False else ans_res['answer']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if ans_res.get("figure"):
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.plotly_chart(ans_res["figure"], use_container_width=True)

        if ans_res.get("data_slice") is not None:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #111111; margin-top: 14px; margin-bottom: 6px;'>Supporting Data Evidence:</div>", unsafe_allow_html=True)
            st.dataframe(ans_res["data_slice"], use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 6: SETTINGS
# ==============================================================================
elif nav_page == "Settings":
    st.markdown(
        """
        <div class="white-card" style="margin-bottom: 1.2rem;">
            <h3 class="card-title-fintech" style="font-size: 1.15rem;">Application Settings & Data Export</h3>
            <div class="card-subtitle-fintech">Configure hygiene rules, optional LLM connectivity, and export sanitized dataset assets.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    s1, s2 = st.columns(2)

    with s1:
        st.markdown("<h4 class='card-title-fintech' style='font-size: 0.95rem; margin-bottom: 0.6rem;'>Data Hygiene Controls</h4>", unsafe_allow_html=True)
        rem_dup = st.checkbox("Automatically drop duplicate rows", value=st.session_state.remove_duplicates)
        if rem_dup != st.session_state.remove_duplicates:
            st.session_state.remove_duplicates = rem_dup
            if st.session_state.current_file_name and st.session_state.raw_df is not None:
                processor = DataProcessor()
                cleaned, summary = processor.clean_dataset(st.session_state.raw_df, remove_duplicates=rem_dup)
                st.session_state.cleaned_df = cleaned
                st.session_state.cleaning_summary = summary
                st.session_state.health_metrics = processor.calculate_health_score(st.session_state.raw_df, cleaned)
                st.success("Cleaning preferences updated.")
                st.rerun()

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 class='card-title-fintech' style='font-size: 0.95rem; margin-bottom: 0.6rem;'>Export Sanitized Dataset</h4>", unsafe_allow_html=True)
        if st.session_state.cleaned_df is not None:
            csv_buffer = io.StringIO()
            st.session_state.cleaned_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Cleaned CSV",
                data=csv_buffer.getvalue(),
                file_name=f"cleaned_{st.session_state.current_file_name or 'dataset.csv'}",
                mime="text/csv",
                use_container_width=True
            )

    with s2:
        st.markdown("<h4 class='card-title-fintech' style='font-size: 0.95rem; margin-bottom: 0.6rem;'>LLM API Integration (Optional)</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.8rem; color: #6B7280; line-height: 1.5;'>NexInsight operates 100% offline with zero external dependencies. You may optionally configure a Gemini API key to synthesize executive leadership briefings.</p>", unsafe_allow_html=True)
        api_key_input = st.text_input("Gemini API Key:", value=st.session_state.llm_api_key, type="password")
        if st.button("Save API Key"):
            st.session_state.llm_api_key = api_key_input
            st.success("API key stored in session state.")

        if st.session_state.llm_api_key and st.session_state.cleaned_df is not None:
            if st.button("Generate Executive Memo via LLM", type="primary", use_container_width=True):
                with st.spinner("Synthesizing executive memo with Gemini..."):
                    memo = InsightEngine.synthesize_with_llm(
                        st.session_state.analysis_results,
                        st.session_state.llm_api_key
                    )
                    if memo:
                        st.markdown("### Executive Memo")
                        st.markdown(memo)
                    else:
                        st.error("Failed to generate memo. Please check your API key.")
