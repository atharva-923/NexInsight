"""
NexInsight — Your AI Data Analyst
Problem Statement 3: The Automated Insight Analyst
Autonomous end-to-end dataset understanding, cleaning, analysis, visualization, and conversational intelligence.
Features multi-file batch upload, complete dataset isolation, and instantaneous dataset switching.
"""

import os
import io
from typing import Dict, Any, List, Optional
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
from core.batch_manager import BatchManager
from core.llm_client import LLMClient, GrokClient
from core.rag_engine import RAGEngine, RAGRetriever

# Load custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "static", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Prevent Streamlit CommonMark 4-space code block bug when rendering indented HTML
_original_st_markdown = st.markdown

def safe_st_markdown(body, *args, **kwargs):
    if isinstance(body, str) and kwargs.get("unsafe_allow_html", False):
        body = " ".join(line.strip() for line in body.splitlines() if line.strip())
    return _original_st_markdown(body, *args, **kwargs)

st.markdown = safe_st_markdown

# Session State Initialization
if "datasets" not in st.session_state:
    st.session_state.datasets = {}  # { dataset_id: dataset_record }
if "active_dataset_id" not in st.session_state:
    st.session_state.active_dataset_id = None
if "batch_processing_logs" not in st.session_state:
    st.session_state.batch_processing_logs = []
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
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.environ.get("GROQ_API_KEY", os.environ.get("XAI_API_KEY", ""))
if "groq_model" not in st.session_state:
    st.session_state.groq_model = os.environ.get("GROQ_MODEL", os.environ.get("XAI_MODEL", LLMClient.DEFAULT_MODEL))
if "llm_api_key" not in st.session_state:
    st.session_state.llm_api_key = st.session_state.groq_api_key
if "xai_api_key" not in st.session_state:
    st.session_state.xai_api_key = st.session_state.groq_api_key
if "xai_model" not in st.session_state:
    st.session_state.xai_model = st.session_state.groq_model
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}  # { dataset_id: list of messages }
if "remove_duplicates" not in st.session_state:
    st.session_state.remove_duplicates = True


def set_active_dataset(dataset_id: str):
    """Activates a dataset in session state and syncs working variables with zero recomputation."""
    if dataset_id in st.session_state.datasets:
        st.session_state.active_dataset_id = dataset_id
        ds = st.session_state.datasets[dataset_id]
        st.session_state.current_file_name = ds["filename"]
        st.session_state.raw_df = ds["raw_df"]
        st.session_state.cleaned_df = ds["cleaned_df"]
        st.session_state.column_types = ds["column_types"]
        st.session_state.cleaning_summary = ds["cleaning_summary"]
        st.session_state.health_metrics = ds["health_metrics"]
        st.session_state.analysis_results = ds["analysis_results"]
        st.session_state.anomalies_data = ds["anomalies_data"]
        st.session_state.ai_insights = ds["ai_insights"]


def load_single_dataset(file_obj, filename: str, set_as_active: bool = True) -> Dict[str, Any]:
    """Ingests and analyzes a single dataset, adding it to the multi-dataset session."""
    existing_names = [d["filename"] for d in st.session_state.datasets.values()]
    res = BatchManager.process_single_file(
        file_obj,
        filename,
        existing_display_names=existing_names,
        remove_duplicates=st.session_state.remove_duplicates
    )
    if res["status"] == "success":
        st.session_state.datasets[res["id"]] = res
        if set_as_active or st.session_state.active_dataset_id is None:
            set_active_dataset(res["id"])
    return res


# Automatically load default Dataset A if no datasets loaded yet
if not st.session_state.datasets:
    default_path = os.path.join("test_datasets", "dataset_a_sales.csv")
    if os.path.exists(default_path):
        load_single_dataset(default_path, "dataset_a_sales.csv", set_as_active=True)
elif st.session_state.active_dataset_id is None or st.session_state.active_dataset_id not in st.session_state.datasets:
    first_id = list(st.session_state.datasets.keys())[0]
    set_active_dataset(first_id)


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.markdown(
    """
    <div class="sidebar-brand-box">
        <div class="brand-title-wrap">
            <h2 class="brand-title">NexInsight</h2>
            <span class="brand-badge">AI ANALYST</span>
        </div>
        <p style="font-size: 0.76rem; color: #6B7280; margin: 0.35rem 0 0 0; font-weight: 500;">Autonomous Data Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<div class='sidebar-nav-title'>Navigation</div>", unsafe_allow_html=True)

nav_page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Explore", "Patterns", "Outliers", "Ask", "Datasets", "Settings"],
    label_visibility="collapsed"
)

# Active Dataset Switcher in Sidebar (when multiple datasets are loaded)
if len(st.session_state.datasets) > 1:
    st.sidebar.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span class='sidebar-nav-title' style='margin: 0;'>Active Dataset</span>
            <span class="pill-badge blue">{len(st.session_state.datasets)} Loaded</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    dataset_opts = {d_id: d["filename"] for d_id, d in st.session_state.datasets.items()}
    cur_id = st.session_state.active_dataset_id if st.session_state.active_dataset_id in dataset_opts else list(dataset_opts.keys())[0]
    
    selected_sidebar_id = st.sidebar.selectbox(
        "Select Active Dataset",
        options=list(dataset_opts.keys()),
        format_func=lambda x: dataset_opts[x],
        index=list(dataset_opts.keys()).index(cur_id),
        key="sidebar_dataset_selector",
        label_visibility="collapsed"
    )
    if selected_sidebar_id != st.session_state.active_dataset_id:
        set_active_dataset(selected_sidebar_id)
        st.rerun()

# Developer / Test Benchmark selector tucked cleanly into an expander
with st.sidebar.expander("Developer / Test Benchmarks", expanded=False):
    sample_options = {
        "Dataset A (Sales & Trends)": "test_datasets/dataset_a_sales.csv",
        "Dataset B (Server Metrics)": "test_datasets/dataset_b_server_metrics.csv",
        "Dataset C (Survey - Dirty)": "test_datasets/dataset_c_survey_dirty.csv",
        "Dataset D (Categorical)": "test_datasets/dataset_d_categorical.csv",
        "Dataset E (Sensors - Numeric)": "test_datasets/dataset_e_sensor_numeric.csv"
    }
    selected_sample = st.selectbox(
        "Quick Switch Dataset",
        list(sample_options.keys()),
        index=0,
        key="sidebar_sample_select"
    )
    if st.button("Load Dataset", use_container_width=True, key="sidebar_load_sample_btn"):
        target_path = sample_options[selected_sample]
        if os.path.exists(target_path):
            filename = os.path.basename(target_path)
            with st.spinner("Processing dataset..."):
                res = load_single_dataset(target_path, filename, set_as_active=True)
            st.rerun()

# Sidebar Active Scope Box
if st.session_state.active_dataset_id and st.session_state.active_dataset_id in st.session_state.datasets:
    active_ds = st.session_state.datasets[st.session_state.active_dataset_id]
    h = active_ds["health_metrics"]
    clean_df = active_ds["cleaned_df"]
    num_c = len([c for c, t in active_ds["column_types"].items() if t == 'Numeric'])
    cat_c = len([c for c, t in active_ds["column_types"].items() if t in ['Categorical', 'Boolean']])
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
                {active_ds['filename']}
            </div>
            <div style="font-size: 0.74rem; color: #6B7280; margin-top: 4px;">
                {len(clean_df):,} rows · {len(clean_df.columns)} attributes<br>
                <span style="color: #9CA3AF;">{num_c} numeric · {cat_c} categorical</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# TOP APP BAR (Light Minimal Header)
# ==============================================================================
active_ds = st.session_state.datasets.get(st.session_state.active_dataset_id, {})
active_display_name = active_ds.get("filename", "No Dataset Loaded")
h_top = active_ds.get("health_metrics", {})
clean_df_top = active_ds.get("cleaned_df")
row_meta = f"{len(clean_df_top):,} rows" if clean_df_top is not None else "0 rows"
col_meta = f"{len(clean_df_top.columns)} attrs" if clean_df_top is not None else "0 attrs"
health_score_val = h_top.get("score", 100) if h_top else 100

batch_pill = f'<div class="dataset-meta-pill"><span style="color: #6B7280;">{len(st.session_state.datasets)} Datasets</span></div>' if len(st.session_state.datasets) > 1 else ''

st.markdown(
    f"""
    <div class="overview-heading-row">
        <div>
            <h1 class="overview-heading">{nav_page}</h1>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <div class="dataset-meta-pill">
                <span class="live-dot"></span>
                <span>Active: <strong style="color: #111111;">{active_display_name}</strong></span>
                <span style="color: #CBD5E1;">·</span>
                <span>{row_meta}</span>
                <span style="color: #CBD5E1;">·</span>
                <span>{col_meta}</span>
                <span style="color: #CBD5E1;">·</span>
                <span class="pill-badge green">{health_score_val}/100</span>
            </div>
            {batch_pill}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# PAGE 1: OVERVIEW (MAIN DASHBOARD)
# ==============================================================================
if nav_page == "Overview":
    if not st.session_state.datasets or not st.session_state.active_dataset_id:
        st.warning("Please upload or select a dataset to begin.")
        st.stop()

    active_ds = st.session_state.datasets[st.session_state.active_dataset_id]
    df = active_ds["cleaned_df"]
    raw_df = active_ds["raw_df"]
    h = active_ds["health_metrics"]
    clean_sum = active_ds["cleaning_summary"]
    anoms = active_ds["anomalies_data"]
    insights = active_ds["ai_insights"]
    col_types = active_ds["column_types"]
    analysis = active_ds["analysis_results"]

    num_cols = [c for c, t in col_types.items() if t == 'Numeric']
    cat_cols = [c for c, t in col_types.items() if t in ['Categorical', 'Boolean']]
    date_cols = [c for c, t in col_types.items() if t == 'Date']

    # 1. Dataset Switcher Ribbon (when multiple datasets are loaded)
    if len(st.session_state.datasets) > 1:
        dataset_opts = {d_id: d["filename"] for d_id, d in st.session_state.datasets.items()}
        cur_id = st.session_state.active_dataset_id
        
        sw_col1, sw_col2 = st.columns([1.5, 2.5])
        with sw_col1:
            st.markdown("<span style='font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #6B7280; letter-spacing: 0.06em;'>Switch Active Dataset:</span>", unsafe_allow_html=True)
            new_active_id = st.selectbox(
                "Active Dataset Switcher",
                options=list(dataset_opts.keys()),
                format_func=lambda x: dataset_opts[x],
                index=list(dataset_opts.keys()).index(cur_id),
                key="overview_dataset_selector",
                label_visibility="collapsed"
            )
            if new_active_id != st.session_state.active_dataset_id:
                set_active_dataset(new_active_id)
                st.rerun()

        with sw_col2:
            st.markdown(
                f"""
                <div style="margin-top: 14px; font-size: 0.8rem; color: #6B7280;">
                    Currently analyzing <strong>{active_ds['filename']}</strong> ({active_ds['cleaned_row_count']:,} rows).
                    Switch to any uploaded file to re-scope all metrics instantly.
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 2. Top 5 Metric Cards (Data Health, Total Records, Attributes, Missing Values, Anomalies)
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
                    <span class="pill-badge gray">{len(df.columns)} Total</span>
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
                    <span class="pill-badge amber">{clean_sum.get('missing_values_handled', 0)} Fixed</span>
                </div>
                <div class="metric-number-lg">{h.get('total_missing', 0):,}</div>
                <div class="metric-subtext-clean">{clean_sum.get('missing_values_handled', 0):,} imputed & resolved</div>
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
                <div class="metric-number-lg" style="color: {'#EF4444' if crit_anom > 0 else '#111111'};">{tot_anom:,}</div>
                <div class="metric-subtext-clean">{crit_anom} critical · {mod_anom} moderate</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 3. Middle Row: Unified Chart Containers (Heading + Chart physically inside the SAME card)
    auto_charts = Visualizer.auto_generate_dashboard_charts(df, col_types)

    chart_col_left, chart_col_right = st.columns([1.75, 1.25])

    with chart_col_left:
        with st.container(border=True):
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                    <div>
                        <h3 class="card-title-fintech" style="font-size: 0.95rem;">Primary Trend / Interaction</h3>
                        <div class="card-subtitle-fintech" style="font-size: 0.78rem;">Dominant temporal trajectory or numerical interaction</div>
                    </div>
                    <span class="pill-badge blue">PRIMARY</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            if auto_charts and len(auto_charts) > 0:
                st.plotly_chart(auto_charts[0], use_container_width=True, key=f"overview_primary_chart_{st.session_state.active_dataset_id}")
            else:
                st.info("No numerical features available for trend analysis.")

    with chart_col_right:
        with st.container(border=True):
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                    <div>
                        <h3 class="card-title-fintech" style="font-size: 0.95rem;">Distribution & Segmentation</h3>
                        <div class="card-subtitle-fintech" style="font-size: 0.78rem;">Categorical share or feature dispersion</div>
                    </div>
                    <span class="pill-badge blue">SEGMENTATION</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            if auto_charts and len(auto_charts) > 1:
                st.plotly_chart(auto_charts[1], use_container_width=True, key=f"overview_secondary_chart_{st.session_state.active_dataset_id}")
            elif auto_charts and len(auto_charts) > 0 and len(col_types) > 1:
                cat_cols = [c for c, t in col_types.items() if t in ['Categorical', 'Boolean']]
                if len(cat_cols) > 1:
                    fig2 = Visualizer.create_donut_chart(df, cat_cols[1])
                    st.plotly_chart(fig2, use_container_width=True, key=f"overview_sec_cat_{st.session_state.active_dataset_id}")
                else:
                    st.info("Primary feature distribution is displayed in the primary panel.")
            else:
                st.info("No categorical or distribution features available.")

    # 4. Bottom Row: 3 White Cards (AI INSIGHTS | DATA QUALITY | ANOMALIES)
    b1, b2, b3 = st.columns(3)

    with b1:
        findings_list = insights.get('key_findings') or ['Attributes evaluated cleanly.']
        patterns_list = insights.get('important_patterns') or ['Distribution is balanced across records.']
        risks_list = insights.get('anomalies') or ['No severe distribution anomalies noted.']
        top_finding = findings_list[0] if findings_list else 'Attributes evaluated cleanly.'
        top_pattern = patterns_list[0] if patterns_list else 'Distribution is balanced across records.'
        top_risk = risks_list[0] if risks_list else 'No severe distribution anomalies noted.'
        st.markdown(
            f"""
            <div class="white-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <h3 class="card-title-fintech">Patterns & Insights</h3>
                    <span class="pill-badge blue">SYNTHESIS</span>
                </div>
                <div class="ai-summary-box">
                    "{insights.get('executive_summary', 'Dataset evaluated with clean metrics.')}"
                </div>
                <div class="insight-sub-card">
                    <div class="insight-sub-tag finding">KEY FINDING</div>
                    <div class="insight-sub-text">{top_finding}</div>
                </div>
                <div class="insight-sub-card">
                    <div class="insight-sub-tag trend">DOMINANT PATTERN</div>
                    <div class="insight-sub-text">{top_pattern}</div>
                </div>
                <div class="insight-sub-card">
                    <div class="insight-sub-tag risk">OBSERVATION</div>
                    <div class="insight-sub-text">{top_risk}</div>
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
        top_reason = top_anom["reason"] if top_anom else "All parameters within standard deviation."

        st.markdown(
            f"""
            <div class="white-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <h3 class="card-title-fintech">Outliers & Anomalies</h3>
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

    # 5. Ask Your Data Prompt Bar at Bottom
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div style="font-size: 0.86rem; font-weight: 600; color: #111111; display: flex; align-items: center; gap: 6px;">
                    <span style="color: #2563EB; font-weight: 800;">▶</span>
                    <span>Ask NexInsight Command Interface</span>
                </div>
                <span class="pill-badge gray">Active: {active_ds['filename']}</span>
            </div>
            <div style="font-size: 0.78rem; color: #6B7280; margin-bottom: 0.6rem;">Direct analytical query execution scoped to active dataset parameters.</div>
            """,
            unsafe_allow_html=True
        )

        dash_q = st.text_input(
            "Ask Your Data",
            placeholder=f"e.g. What is the average {num_cols[0]}? What anomalies exist in {active_ds['filename']}?" if num_cols else "Ask any analytical question...",
            label_visibility="collapsed",
            key="dash_prompt_box"
        )

        if dash_q:
            qa_engine = DataQAEngine(df, col_types, active_ds["analysis_results"], h, dataset_record=active_ds)
            with st.spinner(f"Querying {active_ds['filename']} parameters..."):
                ans_data = qa_engine.answer_query(dash_q, api_key=st.session_state.groq_api_key, model=st.session_state.groq_model)

            highlight_html = f'<span class="pill-badge green">{ans_data["metric_highlight"]}</span>' if ans_data.get("metric_highlight") else ''
            st.markdown(
                f"""
                <div class="ask-data-response">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <span style="font-size: 0.74rem; text-transform: uppercase; color: #2563EB; font-weight: 700; letter-spacing: 0.06em;">Analyst Response</span>
                        {highlight_html}
                    </div>
                    <div style="font-size: 0.9rem; line-height: 1.6; color: #111111;">{ans_data['answer']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if ans_data.get("figure"):
                st.plotly_chart(ans_data["figure"], use_container_width=True, key="overview_prompt_response_fig")


# ==============================================================================
# PAGE 2: EXPLORE (DATA EXPLORER & SCHEMA)
# ==============================================================================
# PAGE 2: EXPLORE (DATA EXPLORER & SCHEMA)
# ==============================================================================
elif nav_page == "Explore":
    if not st.session_state.datasets or not st.session_state.active_dataset_id:
        st.warning("Please upload or select a dataset first.")
        st.stop()

    active_ds = st.session_state.datasets[st.session_state.active_dataset_id]
    df = active_ds["cleaned_df"]
    raw_df = active_ds["raw_df"]
    clean_sum = active_ds["cleaning_summary"]
    h = active_ds["health_metrics"]
    analysis = active_ds["analysis_results"]

    # Health & Hygiene Summary Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='metric-card-fintech'><div class='metric-label-compact'><span>Health Score</span><span class='pill-badge green'>{h.get('status', 'Excellent')}</span></div><div class='metric-number-lg'>{h.get('score', 100)}<span style='font-size: 1rem; color: #9CA3AF;'>/100</span></div><div class='metric-subtext-clean'>Quality score index</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='metric-card-fintech'><div class='metric-label-compact'><span>Duplicates Dropped</span></div><div class='metric-number-lg'>{clean_sum.get('duplicates_removed', 0):,}</div><div class='metric-subtext-clean'>Out of {h.get('total_rows', len(df)):,} rows</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='metric-card-fintech'><div class='metric-label-compact'><span>Missing Imputed</span></div><div class='metric-number-lg'>{clean_sum.get('missing_values_handled', 0):,}</div><div class='metric-subtext-clean'>Statistical imputation</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='metric-card-fintech'><div class='metric-label-compact'><span>Dates Detected</span></div><div class='metric-number-lg'>{clean_sum.get('date_columns_detected', 0)}</div><div class='metric-subtext-clean'>Standardized temporal series</div></div>", unsafe_allow_html=True)

    # Cleaning Audit Trail Accordion
    if clean_sum.get("details"):
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        with st.expander("Data Hygiene Audit Trail", expanded=False):
            for d in clean_sum["details"]:
                st.markdown(f"- {d}")

    # Inferred Schema & Attribute Typings Table
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 class='card-title-fintech' style='margin-bottom: 0.6rem;'>Inferred Schema & Attribute Typings</h3>", unsafe_allow_html=True)
    
    schema_rows = []
    for col, ctype in active_ds["column_types"].items():
        sample_val = str(df[col].iloc[0]) if len(df) > 0 else "N/A"
        unq_cnt = df[col].nunique()
        null_cnt = raw_df[col].isna().sum() if col in raw_df.columns else 0
        schema_rows.append({
            "Attribute Name": col,
            "Type": ctype,
            "Unique Cardinality": f"{unq_cnt:,}",
            "Raw Missing": f"{null_cnt:,}",
            "Sample Representation": sample_val[:45]
        })

    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

    # Cleaned vs Raw Data Sample
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 class='card-title-fintech' style='margin-bottom: 0.6rem;'>Dataset Inspection</h3>", unsafe_allow_html=True)
    view_opt = st.radio("Inspect Data Sample:", ["Cleaned Data", "Raw Data"], horizontal=True)
    if view_opt == "Cleaned Data":
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.dataframe(raw_df.head(100), use_container_width=True)

    # Descriptive Column Statistics Table
    desc_stats = analysis.get("descriptive_stats", {})
    if desc_stats:
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 class='card-title-fintech' style='margin-bottom: 0.6rem;'>Numerical Feature Statistics</h3>", unsafe_allow_html=True)
        stats_df = pd.DataFrame(desc_stats).T
        st.dataframe(stats_df, use_container_width=True)


# ==============================================================================
# PAGE 3: PATTERNS & INSIGHTS
# ==============================================================================
elif nav_page == "Patterns":
    if not st.session_state.datasets or not st.session_state.active_dataset_id:
        st.warning("Please upload or load a dataset first.")
        st.stop()

    active_ds = st.session_state.datasets[st.session_state.active_dataset_id]
    df = active_ds["cleaned_df"]
    insights = active_ds["ai_insights"]
    analysis = active_ds["analysis_results"]

    # Executive Overview
    st.markdown(
        f"""
        <div class="white-card" style="margin-bottom: 1.1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h3 class="card-title-fintech">Executive Overview</h3>
                <span class="pill-badge green">GROUNDED VERIFIED FACTS</span>
            </div>
            <div style="font-size: 0.94rem; line-height: 1.6; color: #111111;">
                "{insights.get('executive_summary', '')}"
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3-Column Findings
    c_find, c_pat, c_obs = st.columns(3)

    with c_find:
        with st.container(border=True):
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <span style="font-weight: 700; font-size: 0.95rem; color: #111111;">Key Findings</span>
                    <span class="pill-badge red">EVIDENCE</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            for idx, item in enumerate(insights.get("key_findings", []), 1):
                st.markdown(
                    f"""
                    <div style="display: flex; gap: 8px; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #F3F4F6;">
                        <span style="font-weight: 800; color: #2563EB; font-size: 0.8rem; font-family: monospace;">{idx:02d}</span>
                        <div style="font-size: 0.82rem; color: #374151; line-height: 1.4;">{item}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with c_pat:
        with st.container(border=True):
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <span style="font-weight: 700; font-size: 0.95rem; color: #111111;">Dominant Patterns</span>
                    <span class="pill-badge blue">TRAJECTORY</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            for idx, item in enumerate(insights.get("important_patterns", []), 1):
                st.markdown(
                    f"""
                    <div style="display: flex; gap: 8px; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #F3F4F6;">
                        <span style="font-weight: 800; color: #2563EB; font-size: 0.8rem; font-family: monospace;">{idx:02d}</span>
                        <div style="font-size: 0.82rem; color: #374151; line-height: 1.4;">{item}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with c_obs:
        with st.container(border=True):
            st.markdown(
                """
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <span style="font-weight: 700; font-size: 0.95rem; color: #111111;">Observations</span>
                    <span class="pill-badge amber">ACTIONABLE</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            for idx, item in enumerate(insights.get("observations", []), 1):
                st.markdown(
                    f"""
                    <div style="display: flex; gap: 8px; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #F3F4F6;">
                        <span style="font-weight: 800; color: #F59E0B; font-size: 0.8rem; font-family: monospace;">{idx:02d}</span>
                        <div style="font-size: 0.82rem; color: #374151; line-height: 1.4;">{item}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Correlation Matrix Section
    corr_data = analysis.get("correlations", {})
    if corr_data.get("has_correlation"):
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 class='card-title-fintech' style='margin-bottom: 0.6rem;'>Cross-Feature Correlations</h3>", unsafe_allow_html=True)
        c_heat, c_pairs = st.columns([1.3, 1])

        with c_heat:
            with st.container(border=True):
                st.plotly_chart(Visualizer.create_correlation_heatmap(corr_data["matrix_df"]), use_container_width=True, key=f"corr_heatmap_{st.session_state.active_dataset_id}")

        with c_pairs:
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-weight: 700; font-size: 0.9rem; color: #111111;">Strongest Interacting Pairs</span>
                        <span class="pill-badge gray">SIGNIFICANT</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                pairs = corr_data.get("significant_pairs", [])
                if pairs:
                    p_df = pd.DataFrame([
                        {"Feature 1": p["col1"], "Feature 2": p["col2"], "Correlation": f"{p['correlation']:.2f}", "Direction": f"{p['strength']} {p['direction']}"}
                        for p in pairs[:8]
                    ])
                    st.dataframe(p_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No significant cross-correlation detected.")

    # Clustering Section
    clustering = analysis.get("clustering")
    if clustering and clustering.get("applicable"):
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <div>
                        <h3 class="card-title-fintech" style="font-size: 0.95rem;">Adaptive K-Means Behavioral Clustering</h3>
                        <div class="card-subtitle-fintech" style="font-size: 0.78rem;">Segmented records into {clustering['n_clusters']} cohesive natural clusters (Silhouette: {clustering['silhouette_score']})</div>
                    </div>
                    <span class="pill-badge gray">{clustering['n_clusters']} CLUSTERS</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            cl_chart, cl_prof = st.columns([1.3, 1])
            with cl_chart:
                st.plotly_chart(Visualizer.create_cluster_scatter(clustering["pca_x"], clustering["pca_y"], clustering["labels"]), use_container_width=True, key=f"cluster_scatter_{st.session_state.active_dataset_id}")

            with cl_prof:
                st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #111111; margin-bottom: 0.5rem;'>Segment Characteristics</div>", unsafe_allow_html=True)
                for p in clustering["profiles"]:
                    st.markdown(
                        f"""
                        <div style="background: #F8FAFC; border: 1px solid #E5E7EB; border-radius: 8px; padding: 0.7rem 0.85rem; margin-bottom: 0.5rem;">
                            <div style="font-weight: 700; color: #2563EB; font-size: 0.85rem;">{p['name']} <span style="font-size: 0.74rem; color: #6B7280; font-weight: 400;">({p['percentage']}% of records)</span></div>
                            <div style="font-size: 0.78rem; color: #374151; margin-top: 2px;">{p['key_traits']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ==============================================================================
# PAGE 4: OUTLIERS & ANOMALIES
# ==============================================================================
elif nav_page == "Outliers":
    if not st.session_state.datasets or not st.session_state.active_dataset_id:
        st.warning("Please upload or load a dataset first.")
        st.stop()

    active_ds = st.session_state.datasets[st.session_state.active_dataset_id]
    df = active_ds["cleaned_df"]
    anoms = active_ds["anomalies_data"]

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
        st.markdown(f"<div class='metric-card-fintech'><div class='metric-label-compact'><span>Low / Boundary</span><span class='pill-badge blue'>Edge</span></div><div class='metric-number-lg' style='color: #3B82F6;'>{low}</div><div class='metric-subtext-clean'>Distribution boundary points</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    anom_list = anoms.get("anomalies_list", [])
    if anom_list:
        # Visual Anomaly Scatter Inspector
        num_cols_with_anoms = list(set([a["column"] for a in anom_list]))
        selected_anom_col = st.selectbox("Inspect Attribute Outliers:", num_cols_with_anoms)
        if selected_anom_col:
            with st.container(border=True):
                st.plotly_chart(Visualizer.create_anomaly_scatter(df, selected_anom_col, anom_list), use_container_width=True, key=f"anomaly_scatter_{st.session_state.active_dataset_id}_{selected_anom_col}")

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
                        <div style="font-size: 1.75rem; font-weight: 800; color: #111111; margin: 0.35rem 0; font-family: 'JetBrains Mono', monospace;">{rec['value']:,}</div>
                        <div style="font-size: 0.76rem; color: #6B7280;">Normal: <strong style="color: #15803D;">{rec['normal_range']}</strong></div>
                        <div style="font-size: 0.74rem; color: #4B5563; margin-top: 0.35rem; line-height: 1.35;">{rec['reason']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Full Anomaly Records Table
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 class='card-title-fintech' style='margin-bottom: 0.6rem;'>Complete Anomaly Register</h3>", unsafe_allow_html=True)
        table_df = pd.DataFrame(anom_list)[["row_index", "column", "value", "normal_range", "z_score", "severity", "reason"]]
        table_df.columns = ["Row #", "Column", "Value", "Normal Range", "Z-Score", "Severity", "Explanation"]
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    else:
        st.success("No anomalies detected in the current numerical parameters.")


# ==============================================================================
# PAGE 5: ASK NEXINSIGHT (AI ANALYST & COMMAND INTERFACE)
# ==============================================================================
elif nav_page == "Ask":
    if not st.session_state.datasets or not st.session_state.active_dataset_id:
        st.warning("Please upload or load a dataset first.")
        st.stop()

    active_ds = st.session_state.datasets[st.session_state.active_dataset_id]
    df = active_ds["cleaned_df"]
    col_types = active_ds["column_types"]
    h = active_ds["health_metrics"]
    active_id = st.session_state.active_dataset_id
    num_cols = [c for c, t in col_types.items() if t == 'Numeric']
    cat_cols = [c for c, t in col_types.items() if t in ['Categorical', 'Boolean']]

    # Isolated chat history for active dataset
    dataset_chat = st.session_state.chat_history.setdefault(active_id, [])

    groq_active = LLMClient.is_configured(st.session_state.groq_api_key)
    groq_status_badge = (
        f'<span class="pill-badge green"><span class="live-dot"></span> Groq AI Active ({st.session_state.groq_model})</span>'
        if groq_active
        else '<span class="pill-badge gray">Local Deterministic Engine Active</span>'
    )

    st.markdown(
        f"""
        <div class="white-card" style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #111111; display: flex; align-items: center; gap: 6px;">
                    <span style="color: #2563EB;">▶</span>
                    <span>Ask NexInsight — Conversational AI Analyst</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    {groq_status_badge}
                    <span class="pill-badge gray">Scope: {active_ds['filename']}</span>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #6B7280;">Grounded Python/Pandas verification + Groq natural-language explanation strictly scoped to {active_ds['filename']} ({active_ds['cleaned_row_count']:,} records).</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Preset Quick Inquiry Chips Generated Dynamically from Actual Dataset Columns
    st.markdown("<p style='font-size: 0.74rem; font-weight: 700; text-transform: uppercase; color: #6B7280; letter-spacing: 0.08em; margin-bottom: 8px;'>Suggested Inquiries for this Dataset:</p>", unsafe_allow_html=True)
    
    sample_queries = []
    if num_cols:
        sample_queries.append(f"What is the average of {num_cols[0]}?")
    if cat_cols:
        sample_queries.append(f"Which {cat_cols[0]} occurs most frequently?")
    sample_queries.append("How many anomalies were detected?")
    if active_ds["analysis_results"].get("correlations", {}).get("has_correlation"):
        sample_queries.append("What are the strongest correlations?")
    sample_queries.append(f"What are the main patterns in {active_ds['filename']}?")

    col_chips = st.columns(min(5, len(sample_queries)))
    selected_chip_query = None
    for idx, (c, q_text) in enumerate(zip(col_chips, sample_queries)):
        with c:
            if st.button(q_text, key=f"chip_{idx}", use_container_width=True):
                selected_chip_query = q_text

    # Chat Conversation History Container
    if dataset_chat:
        with st.container():
            for idx_msg, msg in enumerate(dataset_chat):
                if msg["role"] == "user":
                    st.markdown(f"<div class='chat-bubble-user'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
                elif msg["role"] == "assistant":
                    eng = msg.get("engine", "local")
                    if eng in ["groq", "grok"]:
                        badge_markup = f'<span class="pill-badge blue">Groq AI ({msg.get("model", st.session_state.groq_model)})</span>'
                    elif eng == "fallback":
                        badge_markup = '<span class="pill-badge amber">Local Engine (Groq Fallback)</span>'
                    else:
                        badge_markup = '<span class="pill-badge gray">NexInsight Local Engine</span>'

                    highlight_markup = f'<span class="pill-badge green">{msg["metric_highlight"]}</span>' if msg.get("metric_highlight") else ''

                    st.markdown(
                        f"""
                        <div class="chat-bubble-assistant">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    {badge_markup}
                                    <span style="font-size: 0.74rem; color: #6B7280;">Scoped to {active_ds['filename']}</span>
                                </div>
                                {highlight_markup}
                            </div>
                            <div style="font-size: 0.92rem; line-height: 1.6; color: #111111;">{msg['content']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if msg.get("fallback_message"):
                        st.info(msg["fallback_message"])

                    if msg.get("figure"):
                        st.plotly_chart(msg["figure"], use_container_width=True, key=f"chat_fig_{active_id}_{idx_msg}")

                    if msg.get("data_slice") is not None:
                        st.markdown("<div style='font-size: 0.83rem; font-weight: 700; color: #111111; margin: 8px 0;'>Supporting Data Evidence:</div>", unsafe_allow_html=True)
                        st.dataframe(msg["data_slice"], use_container_width=True, hide_index=True)

    # Input and Action Controls
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    with st.form(key="qa_chat_form", clear_on_submit=True):
        input_cols = st.columns([4, 1])
        with input_cols[0]:
            query_input = st.text_input(
                "Ask a question about the active dataset:",
                value=selected_chip_query or "",
                placeholder=f"e.g. What is the average value in {active_ds['filename']}? What are the key anomalies?",
                label_visibility="collapsed"
            )
        with input_cols[1]:
            submitted = st.form_submit_button("Ask Analyst", type="primary", use_container_width=True)

    action_col1, action_col2 = st.columns([1, 4])
    with action_col1:
        if st.button("Clear Conversation", type="secondary", use_container_width=True):
            st.session_state.chat_history[active_id] = []
            st.rerun()

    # Process query
    if (submitted and query_input) or selected_chip_query:
        actual_query = selected_chip_query if selected_chip_query else query_input
        
        # Append user turn to active dataset history
        dataset_chat.append({"role": "user", "content": actual_query})

        qa_engine = DataQAEngine(
            df,
            col_types,
            active_ds["analysis_results"],
            h,
            dataset_record=active_ds
        )

        with st.spinner(f"Analyzing {active_ds['filename']} parameters & generating grounded response..."):
            ans_res = qa_engine.answer_query(
                actual_query,
                api_key=st.session_state.groq_api_key,
                model=st.session_state.groq_model,
                conversation_history=dataset_chat[:-1]
            )

        # Append assistant turn to active dataset history
        dataset_chat.append({
            "role": "assistant",
            "content": ans_res["answer"],
            "metric_highlight": ans_res.get("metric_highlight"),
            "figure": ans_res.get("figure"),
            "data_slice": ans_res.get("data_slice"),
            "engine": ans_res.get("engine", "local"),
            "model": ans_res.get("model", "NexInsight Engine"),
            "fallback_message": ans_res.get("fallback_message")
        })

        st.rerun()


# ==============================================================================
# PAGE 6: DATASETS & BATCH UPLOAD
# ==============================================================================
elif nav_page == "Datasets":
    st.markdown(
        """
        <div class="white-card" style="margin-bottom: 1.1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <h3 class="card-title-fintech">Batch Dataset Manager</h3>
                <span class="pill-badge blue">MULTI-FILE</span>
            </div>
            <div style="font-size: 0.84rem; color: #6B7280; line-height: 1.5;">
                Upload one or multiple CSV and Excel files simultaneously. Each dataset is processed through the autonomous analytical pipeline with complete isolation and zero recomputation.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Multi-file uploader
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        btn_label = f"Process & Analyze {len(uploaded_files)} Uploaded File{'s' if len(uploaded_files) > 1 else ''}"
        if st.button(btn_label, type="primary", use_container_width=True):
            progress_container = st.empty()
            progress_bar = st.progress(0.0)
            
            stage_weights = {
                "Loading": 0.12,
                "Type Detection": 0.22,
                "Cleaning": 0.35,
                "Statistics": 0.48,
                "Correlation": 0.58,
                "Anomalies": 0.72,
                "Clustering": 0.86,
                "Insights": 0.95,
                "Completed": 1.0
            }

            def ui_progress_callback(file_idx: int, total_files: int, filename: str, stage_name: str, status: str, elapsed: float):
                stage_frac = stage_weights.get(stage_name, 0.5)
                overall_progress = ((file_idx - 1) + stage_frac) / max(1, total_files)
                progress_bar.progress(min(1.0, max(0.0, overall_progress)))
                
                status_text = f"Currently analyzing: <strong>{filename}</strong>"
                stage_text = f"Stage: <strong>{stage_name}</strong>" + (f" ({elapsed:.1f}s)" if elapsed > 0 else "")
                
                progress_container.markdown(
                    f"""
                    <div class="batch-status-card" style="margin-top: 0.8rem; border-left: 3px solid #2563EB;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; color: #111111; font-size: 0.9rem;">Processing File {file_idx}/{total_files}</span>
                            <span class="pill-badge blue">ACTIVE</span>
                        </div>
                        <div style="margin-top: 4px; font-size: 0.85rem; color: #2563EB; font-weight: 600;">
                            {status_text}
                        </div>
                        <div style="font-size: 0.78rem; color: #6B7280; margin-top: 2px;">
                            {stage_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            updated_datasets, logs = BatchManager.process_batch(
                uploaded_files,
                existing_datasets=st.session_state.datasets,
                remove_duplicates=st.session_state.remove_duplicates,
                progress_callback=ui_progress_callback
            )
            st.session_state.datasets = updated_datasets
            st.session_state.batch_processing_logs = logs

            # Set newest successful dataset as active
            for log in reversed(logs):
                if log["status"] == "success":
                    set_active_dataset(log["id"])
                    break

            progress_bar.empty()
            progress_container.empty()
            st.rerun()

    # Processing Status Log Display
    if st.session_state.batch_processing_logs:
        st.markdown("<h4 class='card-title-fintech' style='font-size: 1.05rem; margin-top: 1.2rem; margin-bottom: 0.5rem;'>Processing Telemetry & Stage Timings</h4>", unsafe_allow_html=True)
        for log in st.session_state.batch_processing_logs:
            timings = log.get("stage_timings", {})
            if log["status"] == "success":
                st.markdown(
                    f"""
                    <div class="batch-status-card batch-status-success">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <strong style="color: #111111; font-size: 0.88rem;">{log['filename']}</strong>
                                <span class="pill-badge green">✓ Analyzed</span>
                            </div>
                            <span style="font-size: 0.78rem; font-weight: 700; color: #15803D;">Total: {timings.get('total', 0.0):.2f}s</span>
                        </div>
                        <div style="display: flex; flex-wrap: wrap; gap: 6px 12px; font-size: 0.76rem; padding-top: 6px; border-top: 1px solid #F3F4F6;">
                            <span>Loading: <strong>{timings.get('loading', 0.0):.1f}s</strong></span>
                            <span style="color: #CBD5E1;">·</span>
                            <span>Cleaning: <strong>{timings.get('cleaning', 0.0):.1f}s</strong></span>
                            <span style="color: #CBD5E1;">·</span>
                            <span>Types: <strong>{timings.get('type_detection', 0.0):.1f}s</strong></span>
                            <span style="color: #CBD5E1;">·</span>
                            <span>Stats: <strong>{timings.get('statistics', 0.0):.1f}s</strong></span>
                            <span style="color: #CBD5E1;">·</span>
                            <span>Corr: <strong>{timings.get('correlations', 0.0):.1f}s</strong></span>
                            <span style="color: #CBD5E1;">·</span>
                            <span>Anomalies: <strong>{timings.get('anomalies', 0.0):.1f}s</strong></span>
                            <span style="color: #CBD5E1;">·</span>
                            <span>Clustering: <strong>{timings.get('clustering', 0.0):.1f}s</strong></span>
                            <span style="color: #CBD5E1;">·</span>
                            <span>Insights: <strong>{timings.get('insights', 0.0):.1f}s</strong></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="batch-status-card batch-status-failed">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <strong style="color: #B91C1C; font-size: 0.88rem;">{log['filename']}</strong>
                                <span class="pill-badge red">✗ Ingestion Failed</span>
                            </div>
                            <span style="font-size: 0.78rem; font-weight: 600; color: #EF4444;">Failed</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #B91C1C; margin-top: 4px;">
                            {log.get('message', log.get('error', 'Unknown failure'))}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Session Inventory Table
    if st.session_state.datasets:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h3 class="card-title-fintech" style="font-size: 1.1rem;">Session Inventory ({len(st.session_state.datasets)} Datasets)</h3>
                <span class="pill-badge gray">SCOPED DATASETS</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        batch_summary_df = BatchManager.get_batch_summary_table(st.session_state.datasets)
        st.dataframe(batch_summary_df, use_container_width=True, hide_index=True)

        # Quick Switcher & Reset
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        sw_col_a, sw_col_b = st.columns([2, 1])
        with sw_col_a:
            dataset_opts = {d_id: d["filename"] for d_id, d in st.session_state.datasets.items()}
            cur_id = st.session_state.active_dataset_id
            sel_id = st.selectbox(
                "Change active dataset scope:",
                options=list(dataset_opts.keys()),
                format_func=lambda x: dataset_opts[x],
                index=list(dataset_opts.keys()).index(cur_id) if cur_id in dataset_opts else 0,
                key="datasets_page_selector"
            )
            if sel_id != st.session_state.active_dataset_id:
                set_active_dataset(sel_id)
                st.rerun()

        with sw_col_b:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Clear All Uploaded Datasets", type="secondary", use_container_width=True):
                st.session_state.datasets = {}
                st.session_state.active_dataset_id = None
                st.session_state.batch_processing_logs = []
                st.session_state.chat_history = {}
                st.rerun()


# ==============================================================================
# PAGE 7: SETTINGS
# ==============================================================================
elif nav_page == "Settings":
    st.markdown(
        """
        <div class="white-card" style="margin-bottom: 1.1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <h3 class="card-title-fintech">System Configuration</h3>
                <span class="pill-badge gray">PREFERENCES</span>
            </div>
            <div style="font-size: 0.84rem; color: #6B7280; line-height: 1.5;">
                Configure data hygiene policies, connect xAI Grok API credentials, and export sanitized dataset assets.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    s1, s2 = st.columns(2)

    with s1:
        st.markdown("<h4 class='card-title-fintech' style='font-size: 1.05rem; margin-bottom: 0.6rem;'>Data Hygiene Controls</h4>", unsafe_allow_html=True)
        rem_dup = st.checkbox("Automatically drop duplicate rows", value=st.session_state.remove_duplicates)
        if rem_dup != st.session_state.remove_duplicates:
            st.session_state.remove_duplicates = rem_dup
            processor = DataProcessor()
            for d_id, d in list(st.session_state.datasets.items()):
                cleaned, summary = processor.clean_dataset(d["raw_df"], remove_duplicates=rem_dup)
                d["cleaned_df"] = cleaned
                d["cleaned_row_count"] = len(cleaned)
                d["cleaning_summary"] = summary
                d["health_metrics"] = processor.calculate_health_score(d["raw_df"], cleaned, duplicate_rows_count=summary.get("duplicates_detected"))
            if st.session_state.active_dataset_id:
                set_active_dataset(st.session_state.active_dataset_id)
            st.success("Cleaning preferences updated across session.")
            st.rerun()

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 class='card-title-fintech' style='font-size: 1.05rem; margin-bottom: 0.6rem;'>Export Active Dataset</h4>", unsafe_allow_html=True)
        if st.session_state.active_dataset_id and st.session_state.active_dataset_id in st.session_state.datasets:
            active_ds = st.session_state.datasets[st.session_state.active_dataset_id]
            csv_buffer = io.StringIO()
            active_df = active_ds["cleaned_df"]
            active_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label=f"Download Cleaned CSV ({active_ds['filename']})",
                data=csv_buffer.getvalue(),
                file_name=f"cleaned_{active_ds['filename']}",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 class='card-title-fintech' style='font-size: 1.05rem; margin-bottom: 0.6rem;'>Session Dataset Management</h4>", unsafe_allow_html=True)
        if st.button("Reset / Clear All Session Datasets", type="secondary", use_container_width=True):
            st.session_state.datasets = {}
            st.session_state.active_dataset_id = None
            st.session_state.batch_processing_logs = []
            st.session_state.chat_history = {}
            st.success("Session reset.")
            st.rerun()

    with s2:
        st.markdown("<h4 class='card-title-fintech' style='font-size: 1.05rem; margin-bottom: 0.6rem;'>Groq AI Configuration</h4>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size: 0.82rem; color: #6B7280; line-height: 1.5;'>"
            "NexInsight calculates ground truth deterministically with Python/Pandas. "
            "Connecting the Groq API enables natural-language reasoning, multi-turn dialogue, and executive-level synthesis powered by ultra-fast Llama 3 models."
            "</p>",
            unsafe_allow_html=True
        )

        groq_key_input = st.text_input(
            "Groq API Key (GROQ_API_KEY):",
            value=st.session_state.groq_api_key,
            type="password",
            placeholder="gsk_...",
            help="Your API key is never logged or exposed. It is stored securely in session state."
        )

        groq_model_input = st.text_input(
            "Groq Model (GROQ_MODEL):",
            value=st.session_state.groq_model,
            placeholder=LLMClient.DEFAULT_MODEL,
            help="Default model is 'openai/gpt-oss-120b'. Other models on your Groq key include 'openai/gpt-oss-20b' and 'qwen/qwen3.8-27b'."
        )

        btn_test_col, btn_save_col = st.columns(2)
        with btn_test_col:
            if st.button("Test Groq API", use_container_width=True):
                if not groq_key_input:
                    st.warning("Please enter an API key to test.")
                else:
                    with st.spinner("Pinging Groq endpoint..."):
                        test_res = LLMClient.test_connection(groq_key_input, groq_model_input)
                    if test_res["success"]:
                        st.success(test_res["message"])
                    else:
                        st.error(test_res["error"])

        with btn_save_col:
            if st.button("Save Configuration", type="primary", use_container_width=True):
                st.session_state.groq_api_key = groq_key_input.strip()
                st.session_state.groq_model = groq_model_input.strip() or LLMClient.DEFAULT_MODEL
                # Keep compatibility aliases synced
                st.session_state.xai_api_key = st.session_state.groq_api_key
                st.session_state.xai_model = st.session_state.groq_model
                st.session_state.llm_api_key = st.session_state.groq_api_key
                
                # Write safely to local .env if possible
                try:
                    env_path = os.path.join(os.path.dirname(__file__), ".env")
                    with open(env_path, "w", encoding="utf-8") as f:
                        f.write(f"GROQ_API_KEY={st.session_state.groq_api_key}\n")
                        f.write(f"GROQ_MODEL={st.session_state.groq_model}\n")
                    st.success("Configuration saved to session & .env file.")
                except Exception:
                    st.success("Configuration saved to session.")
                st.rerun()

        # Connection Status Card
        curr_key = st.session_state.groq_api_key
        if curr_key:
            masked_key = curr_key[:6] + "..." + curr_key[-4:] if len(curr_key) > 10 else "***"
            st.markdown(
                f"""
                <div style="background: #F8FAFC; border: 1px solid #E5E7EB; border-left: 3px solid #15803D; border-radius: 8px; padding: 0.85rem 1rem; margin-top: 14px;">
                    <div style="font-size: 0.82rem; font-weight: 700; color: #15803D;">Status: Groq API Key Configured</div>
                    <div style="font-size: 0.76rem; color: #6B7280; margin-top: 2px;">Key: {masked_key} · Model: {st.session_state.groq_model}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="background: #F8FAFC; border: 1px solid #E5E7EB; border-left: 3px solid #9CA3AF; border-radius: 8px; padding: 0.85rem 1rem; margin-top: 14px;">
                    <div style="font-size: 0.82rem; font-weight: 700; color: #111111;">Status: Groq AI Offline (Local Deterministic Active)</div>
                    <div style="font-size: 0.76rem; color: #6B7280; margin-top: 2px;">The system is utilizing 100% local deterministic Python analysis.</div>
                </div>
                """,
                unsafe_allow_html=True
            )



