"""
NexInsight — Your AI Data Analyst
Problem Statement 3: The Automated Insight Analyst
Autonomous end-to-end dataset understanding, cleaning, analysis, visualization, and conversational intelligence.
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
    <div style="padding: 0.8rem 0 1.2rem 0; border-bottom: 1px solid rgba(255,255,255,0.08);">
        <h2 style="font-size: 1.25rem; font-weight: 700; margin: 0; color: #FFFFFF; letter-spacing: -0.02em;">NexInsight</h2>
        <p style="font-size: 0.78rem; color: #94A3B8; margin: 0.2rem 0 0 0;">Your AI Data Analyst</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

nav_page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Data Upload", "Insights", "Anomalies", "AI Analyst", "Settings"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Quick Dataset Switcher for Demo
st.sidebar.markdown("<p style='font-size: 0.78rem; font-weight: 600; text-transform: uppercase; color: #64748B; letter-spacing: 0.05em; margin-bottom: 8px;'>Instant Test Datasets</p>", unsafe_allow_html=True)
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

if st.sidebar.button("Load Selected Dataset", use_container_width=True):
    target_path = sample_options[selected_sample]
    if os.path.exists(target_path):
        filename = os.path.basename(target_path)
        process_dataset(target_path, filename)
        st.rerun()

# Sidebar Active Dataset Info
if st.session_state.current_file_name:
    h = st.session_state.health_metrics
    st.sidebar.markdown(
        f"""
        <div style="margin-top: 1.5rem; padding: 0.9rem; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;">
            <div style="font-size: 0.72rem; text-transform: uppercase; color: #64748B; font-weight: 600;">Active Dataset</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #F1F5F9; margin-top: 2px; word-break: break-all;">{st.session_state.current_file_name}</div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">Health: <span style="color: {h.get('status_color', '#10B981')}; font-weight: 600;">{h.get('score', 0)}/100</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )


# TOP APP HEADER
st.markdown(
    f"""
    <div class="app-header">
        <div>
            <h1 class="brand-title">NexInsight — Your AI Data Analyst</h1>
            <div class="brand-subtitle">Automated Data Understanding, Cleaning, Statistical Analysis & Strategic Insights</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Current Dataset</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC;">{st.session_state.current_file_name or "No File Loaded"}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if nav_page == "Dashboard":
    if st.session_state.cleaned_df is None:
        st.warning("Please upload or load a dataset first.")
        st.stop()

    df = st.session_state.cleaned_df
    raw_df = st.session_state.raw_df
    h = st.session_state.health_metrics
    clean_sum = st.session_state.cleaning_summary
    anoms = st.session_state.anomalies_data
    insights = st.session_state.ai_insights

    # Top KPI Metrics Row
    m1, m2, m3, m4, m5 = st.columns([1.2, 1, 1, 1, 1])

    with m1:
        st.markdown(
            f"""
            <div class="health-box">
                <div class="health-status">{h.get('status', 'Optimal')} Quality</div>
                <div class="health-val">{h.get('score', 100)}<span style="font-size: 1.1rem; color: #94A3B8;">/100</span></div>
                <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 2px;">Dataset Health Score</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Total Rows</div>
                <div class="metric-card-value">{h.get('total_rows', len(df)):,}</div>
                <div class="metric-card-sub">{len(df):,} cleaned records</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Attributes</div>
                <div class="metric-card-value">{h.get('total_columns', len(df.columns))}</div>
                <div class="metric-card-sub">{len([c for c, t in st.session_state.column_types.items() if t == 'Numeric'])} numeric, {len([c for c, t in st.session_state.column_types.items() if t in ['Categorical', 'Boolean']])} cat</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Missing Values</div>
                <div class="metric-card-value">{h.get('total_missing', 0):,}</div>
                <div class="metric-card-sub">{clean_sum.get('missing_values_handled', 0):,} handled</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-label">Anomalies Detected</div>
                <div class="metric-card-value">{anoms.get('total_anomalies', 0):,}</div>
                <div class="metric-card-sub">{clean_sum.get('duplicates_removed', 0):,} duplicates removed</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Executive Summary & Anomaly Banner
    col_ai, col_anom = st.columns([2, 1])

    with col_ai:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="card-title">Executive AI Summary</div>
                <div class="card-description">Factual narrative synthesized directly from live computed dataset statistics.</div>
                <div style="font-size: 0.92rem; line-height: 1.6; color: #F1F5F9;">
                    {insights.get('executive_summary', 'Analysis completed successfully.')}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_anom:
        tot_anom = anoms.get("total_anomalies", 0)
        crit_anom = anoms.get("severity_counts", {}).get("Critical", 0)
        st.markdown(
            f"""
            <div class="dashboard-card" style="border-left: 4px solid {'#EF4444' if crit_anom > 0 else '#10B981'};">
                <div class="card-title">Risk & Anomaly Monitor</div>
                <div class="card-description">Real-time distribution outlier assessment.</div>
                <div style="margin-top: 0.5rem;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: {'#EF4444' if crit_anom > 0 else '#10B981'};">{tot_anom} Outliers Flagged</div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">Critical severity: <strong style="color: #F8FAFC;">{crit_anom}</strong> | Moderate: <strong style="color: #F8FAFC;">{anoms.get('severity_counts', {}).get('Moderate', 0)}</strong></div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 6px;">Check the Anomalies tab for full record inspection.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Key Analytical Findings Highlights
    st.markdown("<div class='card-title' style='margin-bottom: 0.8rem;'>Key Strategic Findings</div>", unsafe_allow_html=True)
    f_cols = st.columns(3)
    findings = insights.get("key_findings", []) + insights.get("important_patterns", [])
    for idx, col in enumerate(f_cols):
        if idx < len(findings):
            with col:
                st.markdown(
                    f"""
                    <div class="insight-item insight-item-highlight">
                        {findings[idx]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Automatically Generated Charts Grid
    st.markdown("<div class='card-title' style='margin-bottom: 0.4rem;'>Automated Interactive Visualizations</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.82rem; color: #94A3B8; margin-bottom: 1.2rem;'>Charts dynamically generated based on detected column types and meaningful distributions.</p>", unsafe_allow_html=True)

    auto_charts = Visualizer.auto_generate_dashboard_charts(df, st.session_state.column_types)

    if auto_charts:
        for i in range(0, len(auto_charts), 2):
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(auto_charts[i], use_container_width=True)
            if i + 1 < len(auto_charts):
                with c2:
                    st.plotly_chart(auto_charts[i + 1], use_container_width=True)
    else:
        st.info("No numerical or categorical data available to construct visualizations.")

    # Quick Ask Your Data Console
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="dashboard-card">
            <div class="card-title">Ask Your Data</div>
            <div class="card-description">Ask any plain-English question about the active dataset.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    dash_q = st.text_input("Enter your analytical question (e.g. 'What is the highest-performing category?', 'Are there unusual records?')", key="dash_question")
    if dash_q:
        qa_engine = DataQAEngine(
            df,
            st.session_state.column_types,
            st.session_state.analysis_results,
            st.session_state.health_metrics
        )
        ans_data = qa_engine.answer_query(dash_q)
        st.markdown(
            f"""
            <div class="qa-console-response">
                <div style="font-size: 0.78rem; text-transform: uppercase; color: #6366F1; font-weight: 600; margin-bottom: 0.4rem;">NexInsight Intelligence</div>
                <div class="qa-answer-text">{ans_data['answer']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if ans_data.get("figure"):
            st.plotly_chart(ans_data["figure"], use_container_width=True)


# ==============================================================================
# PAGE 2: DATA UPLOAD
# ==============================================================================
elif nav_page == "Data Upload":
    st.markdown("<div class='card-title'>Dataset Ingestion & Schema Inspector</div>", unsafe_allow_html=True)
    st.markdown("<p class='card-description'>Upload an unseen CSV or XLSX file. NexInsight will automatically inspect schema, perform hygiene checks, and quantify health.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    if uploaded_file is not None:
        if st.button("Process & Analyze Uploaded Dataset", type="primary"):
            process_dataset(uploaded_file, uploaded_file.name)
            st.success(f"Successfully processed {uploaded_file.name}")
            st.rerun()

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df
        raw_df = st.session_state.raw_df
        clean_sum = st.session_state.cleaning_summary
        h = st.session_state.health_metrics

        # Health & Cleaning Overview Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"<div class='metric-card'><div class='metric-card-label'>Health Score</div><div class='metric-card-value' style='color: {h.get('status_color', '#10B981')};'>{h.get('score', 100)}/100</div><div class='metric-card-sub'>{h.get('status', 'Optimal')}</div></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='metric-card'><div class='metric-card-label'>Duplicates Removed</div><div class='metric-card-value'>{clean_sum.get('duplicates_removed', 0):,}</div><div class='metric-card-sub'>Out of {h.get('total_rows', len(df)):,} total rows</div></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='metric-card'><div class='metric-card-label'>Missing Values Handled</div><div class='metric-card-value'>{clean_sum.get('missing_values_handled', 0):,}</div><div class='metric-card-sub'>Imputed via median/mode</div></div>", unsafe_allow_html=True)
        with k4:
            st.markdown(f"<div class='metric-card'><div class='metric-card-label'>Dates Detected</div><div class='metric-card-value'>{clean_sum.get('date_columns_detected', 0)}</div><div class='metric-card-sub'>Standardized temporal series</div></div>", unsafe_allow_html=True)

        # Cleaning Audit Trail
        if clean_sum.get("details"):
            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            with st.expander("Data Cleaning Audit Summary", expanded=True):
                for d in clean_sum["details"]:
                    st.markdown(f"- {d}")

        # Detected Column Schema Table
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Automatic Schema & Column Type Detection</div>", unsafe_allow_html=True)
        
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
                "Sample Value": sample_val[:40]
            })

        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

        # Raw vs Cleaned Preview Toggle
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        view_opt = st.radio("Preview Dataset:", ["Cleaned Data", "Raw Data"], horizontal=True)
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

    st.markdown("<div class='card-title'>Strategic & AI-Generated Insights</div>", unsafe_allow_html=True)
    st.markdown("<p class='card-description'>Comprehensive analytical synthesis based strictly on computed dataset attributes.</p>", unsafe_allow_html=True)

    # Executive Narrative Card
    st.markdown(
        f"""
        <div class="dashboard-card">
            <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: #6366F1; letter-spacing: 0.05em; margin-bottom: 0.3rem;">Executive Summary</div>
            <div style="font-size: 1.05rem; line-height: 1.6; color: #F8FAFC;">{insights.get('executive_summary', '')}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3-Column Structured Breakdown
    c_find, c_pat, c_obs = st.columns(3)

    with c_find:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #F1F5F9; margin-bottom: 0.6rem;'>Key Statistical Findings</div>", unsafe_allow_html=True)
        for item in insights.get("key_findings", []):
            st.markdown(f"<div class='insight-item'>{item}</div>", unsafe_allow_html=True)

    with c_pat:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #F1F5F9; margin-bottom: 0.6rem;'>Important Patterns</div>", unsafe_allow_html=True)
        for item in insights.get("important_patterns", []):
            st.markdown(f"<div class='insight-item insight-item-highlight'>{item}</div>", unsafe_allow_html=True)

    with c_obs:
        st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #F1F5F9; margin-bottom: 0.6rem;'>Strategic Observations</div>", unsafe_allow_html=True)
        for item in insights.get("observations", []):
            st.markdown(f"<div class='insight-item insight-item-success'>{item}</div>", unsafe_allow_html=True)

    # Correlation Matrix Section
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    corr_data = analysis.get("correlations", {})
    if corr_data.get("has_correlation"):
        st.markdown("<div class='card-title'>Correlation Analysis & Interdependencies</div>", unsafe_allow_html=True)
        c_heat, c_pairs = st.columns([1.3, 1])

        with c_heat:
            st.plotly_chart(Visualizer.create_correlation_heatmap(corr_data["matrix_df"]), use_container_width=True)

        with c_pairs:
            st.markdown("<div style='font-size: 0.9rem; font-weight: 600; color: #F1F5F9; margin-bottom: 0.5rem;'>Strongest Attribute Pairs</div>", unsafe_allow_html=True)
            pairs = corr_data.get("significant_pairs", [])
            if pairs:
                p_df = pd.DataFrame([
                    {"Feature 1": p["col1"], "Feature 2": p["col2"], "Correlation": f"{p['correlation']:.2f}", "Relationship": f"{p['strength']} {p['direction']}"}
                    for p in pairs[:8]
                ])
                st.dataframe(p_df, use_container_width=True, hide_index=True)
            else:
                st.info("No significant cross-correlation detected between numerical attributes.")

    # Clustering Section
    clustering = analysis.get("clustering")
    if clustering and clustering.get("applicable"):
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Adaptive K-Means Behavioral Clustering</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='card-description'>Partitioned records into {clustering['n_clusters']} natural cohesive clusters based on standardized numerical properties.</p>", unsafe_allow_html=True)

        cl_chart, cl_prof = st.columns([1.3, 1])
        with cl_chart:
            st.plotly_chart(Visualizer.create_cluster_scatter(clustering["pca_x"], clustering["pca_y"], clustering["labels"]), use_container_width=True)

        with cl_prof:
            st.markdown("<div style='font-size: 0.9rem; font-weight: 600; color: #F1F5F9; margin-bottom: 0.5rem;'>Segment Profiles</div>", unsafe_allow_html=True)
            for p in clustering["profiles"]:
                st.markdown(
                    f"""
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 0.8rem; margin-bottom: 0.6rem;">
                        <div style="font-weight: 700; color: #6366F1;">{p['name']} <span style="font-size: 0.8rem; color: #94A3B8; font-weight: 400;">({p['percentage']}% of records)</span></div>
                        <div style="font-size: 0.8rem; color: #E2E8F0; margin-top: 4px;">Key Traits: {p['key_traits']}</div>
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

    st.markdown("<div class='card-title'>Anomaly & Statistical Outlier Intelligence</div>", unsafe_allow_html=True)
    st.markdown("<p class='card-description'>Automated outlier detection using Interquartile Range (IQR) and Z-score distribution thresholds.</p>", unsafe_allow_html=True)

    # Anomaly KPI Cards
    a1, a2, a3, a4 = st.columns(4)
    tot = anoms.get("total_anomalies", 0)
    crit = anoms.get("severity_counts", {}).get("Critical", 0)
    mod = anoms.get("severity_counts", {}).get("Moderate", 0)
    low = anoms.get("severity_counts", {}).get("Low", 0)

    with a1:
        st.markdown(f"<div class='metric-card'><div class='metric-card-label'>Total Anomalies</div><div class='metric-card-value'>{tot:,}</div><div class='metric-card-sub'>Flagged across dataset</div></div>", unsafe_allow_html=True)
    with a2:
        st.markdown(f"<div class='metric-card'><div class='metric-card-label'>Critical Severity</div><div class='metric-card-value' style='color: #EF4444;'>{crit}</div><div class='metric-card-sub'>Z-score > 3.5</div></div>", unsafe_allow_html=True)
    with a3:
        st.markdown(f"<div class='metric-card'><div class='metric-card-label'>Moderate Severity</div><div class='metric-card-value' style='color: #F59E0B;'>{mod}</div><div class='metric-card-sub'>Outside 1.5x IQR</div></div>", unsafe_allow_html=True)
    with a4:
        st.markdown(f"<div class='metric-card'><div class='metric-card-label'>Low / Boundary</div><div class='metric-card-value' style='color: #3B82F6;'>{low}</div><div class='metric-card-sub'>Distribution edge</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    anom_list = anoms.get("anomalies_list", [])
    if anom_list:
        # Visual Anomaly Inspector
        num_cols_with_anoms = list(set([a["column"] for a in anom_list]))
        selected_anom_col = st.selectbox("Inspect Attribute Outliers:", num_cols_with_anoms)
        if selected_anom_col:
            st.plotly_chart(Visualizer.create_anomaly_scatter(df, selected_anom_col, anom_list), use_container_width=True)

        # Highlight Top 3 Most Extreme Anomalies in Cards (as specified in Problem Statement)
        st.markdown("<div class='card-title'>Most Severe Outlier Records</div>", unsafe_allow_html=True)
        top_cards_cols = st.columns(min(3, len(anom_list)))
        for i, c in enumerate(top_cards_cols):
            rec = anom_list[i]
            badge_cls = "badge-critical" if rec["severity"] == "Critical" else "badge-moderate"
            with c:
                st.markdown(
                    f"""
                    <div class="dashboard-card" style="border-top: 3px solid {'#EF4444' if rec['severity'] == 'Critical' else '#F59E0B'};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 0.78rem; text-transform: uppercase; color: #94A3B8; font-weight: 600;">{rec['column']}</span>
                            <span class="{badge_cls}">{rec['severity']}</span>
                        </div>
                        <div style="font-size: 1.6rem; font-weight: 700; color: #F8FAFC; margin: 0.5rem 0;">{rec['value']:,}</div>
                        <div style="font-size: 0.8rem; color: #94A3B8;">Normal Range: <span style="color: #10B981; font-weight: 600;">{rec['normal_range']}</span></div>
                        <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.4rem;">{rec['reason']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Full Anomaly Table
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>All Detected Anomalies</div>", unsafe_allow_html=True)
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

    st.markdown("<div class='card-title'>AI Analyst — Ask Your Data</div>", unsafe_allow_html=True)
    st.markdown("<p class='card-description'>Interact with your dataset using natural language. Questions are resolved against live computed statistics without hardcoded rules.</p>", unsafe_allow_html=True)

    # Preset Quick Inquiry Chips
    st.markdown("<p style='font-size: 0.82rem; font-weight: 600; color: #94A3B8; margin-bottom: 8px;'>Suggested Inquiries:</p>", unsafe_allow_html=True)
    
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
            <div class="qa-console-response">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <span style="font-size: 0.78rem; text-transform: uppercase; color: #6366F1; font-weight: 700; letter-spacing: 0.05em;">AI Data Analyst Response</span>
                    {f'<span class="badge-low">{ans_res["metric_highlight"]}</span>' if ans_res.get("metric_highlight") else ''}
                </div>
                <div class="qa-answer-text">{ans_res['answer']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if ans_res.get("figure"):
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.plotly_chart(ans_res["figure"], use_container_width=True)

        if ans_res.get("data_slice") is not None:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #F1F5F9; margin-top: 12px; margin-bottom: 6px;'>Supporting Data Slice:</div>", unsafe_allow_html=True)
            st.dataframe(ans_res["data_slice"], use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 6: SETTINGS
# ==============================================================================
elif nav_page == "Settings":
    st.markdown("<div class='card-title'>Application Settings & Export</div>", unsafe_allow_html=True)
    st.markdown("<p class='card-description'>Configure cleaning rules, external LLM integration, and export sanitized data.</p>", unsafe_allow_html=True)

    s1, s2 = st.columns(2)

    with s1:
        st.markdown("<div class='card-title' style='font-size: 0.95rem;'>Data Hygiene Controls</div>", unsafe_allow_html=True)
        rem_dup = st.checkbox("Automatically drop duplicate rows", value=st.session_state.remove_duplicates)
        if rem_dup != st.session_state.remove_duplicates:
            st.session_state.remove_duplicates = rem_dup
            if st.session_state.current_file_name and st.session_state.raw_df is not None:
                # Re-clean dataset with updated rule
                processor = DataProcessor()
                cleaned, summary = processor.clean_dataset(st.session_state.raw_df, remove_duplicates=rem_dup)
                st.session_state.cleaned_df = cleaned
                st.session_state.cleaning_summary = summary
                st.session_state.health_metrics = processor.calculate_health_score(st.session_state.raw_df, cleaned)
                st.success("Cleaning preferences updated.")
                st.rerun()

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='card-title' style='font-size: 0.95rem;'>Export Cleaned Dataset</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='card-title' style='font-size: 0.95rem;'>LLM API Integration (Optional)</div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.8rem; color: #94A3B8;'>NexInsight operates 100% offline with zero external dependencies. You may optionally configure a Gemini API key to synthesize executive leadership briefings.</p>", unsafe_allow_html=True)
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
