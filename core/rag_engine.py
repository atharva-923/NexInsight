"""
NexInsight - Tabular RAG & Structured Knowledge Retrieval Engine
Builds compact, grounded analytical knowledge representations of the active dataset
and retrieves only the strictly relevant sections based on user inquiries.
"""

import re
import difflib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Set


class RAGEngine:
    """Lightweight, structured RAG engine for tabular data intelligence."""

    @staticmethod
    def get_system_prompt(filename: str) -> str:
        """Returns strict system prompt instructions for the Groq AI Analyst."""
        return (
            "You are NexInsight AI Analyst, an expert autonomous data analyst assistant.\n"
            f"You are analyzing the ACTIVE dataset: '{filename}'.\n\n"
            "STRICT RULES:\n"
            "1. Answer ONLY using the supplied verified dataset context and Python calculations.\n"
            "2. Do NOT invent numbers, percentages, counts, metrics, or column names.\n"
            "3. Do NOT claim calculations that were not provided in the context.\n"
            "4. Python / NexInsight calculations are the authoritative source of truth. Always quote and agree with them.\n"
            "5. If the provided context is insufficient to answer the question with certainty, clearly state:\n"
            "   'The available dataset context is insufficient to answer this question reliably.'\n"
            "6. Do NOT pretend to have access to the raw full dataset beyond the supplied summary context.\n"
            "7. Explain results clearly, concisely, and professionally in 2 to 4 sentences.\n"
            "8. Always mention the relevant column name, category, or time period when explaining findings.\n"
            "9. Do not provide unsupported conclusions or speculative claims."
        )

    @classmethod
    def _extract_mentioned_columns(cls, query: str, columns: List[str]) -> List[str]:
        """Identifies columns mentioned in the query using exact and fuzzy matching."""
        q_lower = query.lower()
        mentioned: List[str] = []
        for c in columns:
            if c.lower() in q_lower:
                mentioned.append(c)
        if not mentioned:
            words = re.findall(r"\w+", q_lower)
            for w in words:
                if len(w) <= 2:
                    continue
                matches = difflib.get_close_matches(w, [c.lower() for c in columns], n=1, cutoff=0.75)
                if matches:
                    for col in columns:
                        if col.lower() == matches[0] and col not in mentioned:
                            mentioned.append(col)
        return mentioned

    @classmethod
    def build_dataset_knowledge_sections(
        cls,
        active_dataset: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Builds the 8 core structured knowledge sections for the active dataset.
        Zero fake data. Only real metrics computed by NexInsight.
        """
        filename = active_dataset.get("filename", "active_dataset.csv")
        cleaned_rows = active_dataset.get("cleaned_row_count", 0)
        col_count = active_dataset.get("column_count", 0)
        col_types = active_dataset.get("column_types", {})
        h = active_dataset.get("health_metrics", {})
        clean_sum = active_dataset.get("cleaning_summary", {})
        analysis = active_dataset.get("analysis_results", {})
        anoms_data = active_dataset.get("anomalies_data", {})
        insights = active_dataset.get("ai_insights", {})
        raw_df = active_dataset.get("raw_df")
        df = active_dataset.get("cleaned_df")

        sections: Dict[str, str] = {}

        # 1. DATASET PROFILE
        type_breakdown = []
        for c, t in col_types.items():
            type_breakdown.append(f"{c} ({t})")
        sections["profile"] = (
            f"=== SECTION 1: DATASET PROFILE ===\n"
            f"Filename: {filename}\n"
            f"Total Records: {cleaned_rows:,} cleaned rows (raw entries: {len(raw_df) if raw_df is not None else cleaned_rows:,})\n"
            f"Attributes: {col_count} columns\n"
            f"Attributes List & Detected Types: {', '.join(type_breakdown)}"
        )

        # 2. DATA QUALITY
        missing_by_col = []
        if raw_df is not None:
            for col in raw_df.columns:
                n_miss = int(raw_df[col].isna().sum())
                if n_miss > 0:
                    missing_by_col.append(f"{col}: {n_miss:,} missing ({n_miss / len(raw_df) * 100:.1f}%)")
        missing_str = "; ".join(missing_by_col) if missing_by_col else "Zero missing values detected across all columns."

        sections["quality"] = (
            f"=== SECTION 2: DATA QUALITY & HYGIENE ===\n"
            f"Health Score: {h.get('score', 100)}/100 (Rating: {h.get('status', 'Excellent')})\n"
            f"Missing Values: {h.get('total_missing', 0):,} total missing values in raw dataset ({clean_sum.get('missing_values_handled', 0):,} resolved)\n"
            f"Missing Values Breakdown by Column: {missing_str}\n"
            f"Duplicates: {clean_sum.get('duplicates_removed', 0):,} duplicate rows dropped\n"
            f"Completeness Index: {h.get('completeness_pct', 100.0)}%\n"
            f"Uniqueness Index: {h.get('uniqueness_pct', 100.0)}%\n"
            f"Structure Validity: {h.get('validity_pct', 100.0)}%"
        )

        # 3. NUMERIC STATISTICS
        desc_stats = analysis.get("descriptive_stats", {})
        num_stat_lines = []
        if desc_stats:
            for col, stats in desc_stats.items():
                mean_val = stats.get('mean', 0.0)
                std_val = stats.get('std', 0.0)
                min_val = stats.get('min', 0.0)
                median_val = stats.get('50%', stats.get('median', 0.0))
                max_val = stats.get('max', 0.0)
                num_stat_lines.append(
                    f"• {col}: Count={int(stats.get('count', cleaned_rows)):,}, Mean={mean_val:,.2f}, "
                    f"Median={median_val:,.2f}, StdDev={std_val:,.2f}, Min={min_val:,.2f}, Max={max_val:,.2f}"
                )
        sections["numeric"] = (
            f"=== SECTION 3: NUMERIC STATISTICS ===\n"
            + ("\n".join(num_stat_lines) if num_stat_lines else "No numerical attributes available.")
        )

        # 4. CATEGORICAL INFORMATION
        cat_cols = [c for c, t in col_types.items() if t in ['Categorical', 'Boolean']]
        cat_lines = []
        if df is not None and cat_cols:
            for col in cat_cols[:8]:  # Top 8 categorical features
                unq = df[col].nunique()
                vc = df[col].value_counts(dropna=False).head(3)
                top_items = [f"'{k}' ({v:,} occurrences, {v / len(df) * 100:.1f}%)" for k, v in vc.items()]
                cat_lines.append(f"• {col} ({unq:,} unique values): Top categories: {', '.join(top_items)}")
        sections["categorical"] = (
            f"=== SECTION 4: CATEGORICAL INFORMATION ===\n"
            + ("\n".join(cat_lines) if cat_lines else "No categorical attributes available.")
        )

        # 5. RELATIONSHIPS & CORRELATIONS
        corr_data = analysis.get("correlations", {})
        rel_lines = []
        if corr_data.get("has_correlation"):
            pairs = corr_data.get("significant_pairs", [])
            for p in pairs[:6]:
                rel_lines.append(
                    f"• {p['col1']} ↔ {p['col2']}: Pearson r = {p['correlation']:.2f} ({p['strength']} {p['direction']} correlation)"
                )
        sections["relationships"] = (
            f"=== SECTION 5: RELATIONSHIPS & CORRELATIONS ===\n"
            + ("\n".join(rel_lines) if rel_lines else "No strong linear correlations detected between numerical parameters.")
        )

        # 6. TEMPORAL PATTERNS
        date_cols = [c for c, t in col_types.items() if t == 'Date']
        temporal_lines = []
        if date_cols and df is not None:
            for d_col in date_cols:
                try:
                    s_date = pd.to_datetime(df[d_col], errors='coerce').dropna()
                    if not s_date.empty:
                        temporal_lines.append(
                            f"• Temporal Axis '{d_col}': Span from {s_date.min().strftime('%Y-%m-%d')} to {s_date.max().strftime('%Y-%m-%d')} ({s_date.nunique():,} unique dates)"
                        )
                except Exception:
                    pass
        # Add any trend analysis from analysis_results
        trend_info = analysis.get("trends", {})
        if trend_info and isinstance(trend_info, dict):
            for t_k, t_v in trend_info.items():
                temporal_lines.append(f"• Trend ({t_k}): {t_v}")

        sections["temporal"] = (
            f"=== SECTION 6: TEMPORAL PATTERNS & TRENDS ===\n"
            + ("\n".join(temporal_lines) if temporal_lines else "No time series or date columns detected.")
        )

        # 7. ANOMALIES & OUTLIERS
        tot_anom = anoms_data.get("total_anomalies", 0)
        sev = anoms_data.get("severity_counts", {})
        anom_list = anoms_data.get("anomalies_list", [])
        anom_lines = [
            f"Total Anomalies Flagged: {tot_anom:,} (Critical: {sev.get('Critical', 0):,}, Moderate: {sev.get('Moderate', 0):,}, Low: {sev.get('Low', 0):,})"
        ]
        if anom_list:
            anom_lines.append("Most Severe Outliers:")
            for a in anom_list[:5]:
                anom_lines.append(
                    f"• Row {a['row_index']} in '{a['column']}': Value={a['value']:,} (Expected Normal: {a['normal_range']}, Z-Score: {a['z_score']:.1f}, Severity: {a['severity']}). {a['reason']}"
                )
        sections["anomalies"] = (
            f"=== SECTION 7: ANOMALIES & OUTLIERS ===\n"
            + "\n".join(anom_lines)
        )

        # 8. EXISTING INSIGHTS
        findings = insights.get("key_findings", [])
        patterns = insights.get("important_patterns", [])
        obs = insights.get("observations", [])
        exec_sum = insights.get("executive_summary", "")
        insight_lines = []
        if exec_sum:
            insight_lines.append(f"Executive Summary: \"{exec_sum}\"")
        if findings:
            insight_lines.append("Key Findings: " + " | ".join(findings[:4]))
        if patterns:
            insight_lines.append("Dominant Patterns: " + " | ".join(patterns[:3]))
        if obs:
            insight_lines.append("Observations & Risks: " + " | ".join(obs[:3]))

        sections["insights"] = (
            f"=== SECTION 8: GENERATED INSIGHTS ===\n"
            + ("\n".join(insight_lines) if insight_lines else "Standard dataset baseline evaluation.")
        )

        return sections

    @classmethod
    def retrieve_relevant_context(
        cls,
        query: str,
        active_dataset: Dict[str, Any],
        verified_result: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Selectively retrieves ONLY the relevant sections for the query.
        Keeps Groq token usage minimal, compact, and strictly grounded.
        """
        all_sections = cls.build_dataset_knowledge_sections(active_dataset)
        q_lower = query.lower()
        col_types = active_dataset.get("column_types", {})
        all_cols = list(col_types.keys())
        mentioned_cols = cls._extract_mentioned_columns(query, all_cols)

        # Determine which sections to include
        selected_keys: Set[str] = {"profile"}  # Profile is always included for context

        # 1. Quality & Missing values
        if any(w in q_lower for w in ["missing", "null", "clean", "hygiene", "quality", "health", "duplicate", "error", "nan"]):
            selected_keys.add("quality")

        # 2. Anomalies & Outliers
        if any(w in q_lower for w in ["anomal", "outlier", "spike", "unusual", "abnormal", "extreme", "deviat", "weird"]):
            selected_keys.add("anomalies")

        # 3. Correlations & Relationships
        if any(w in q_lower for w in ["correlat", "relation", "dependency", "interact", "pair", "between", "associated", "impact", "affect"]):
            selected_keys.add("relationships")

        # 4. Temporal & Trends
        if any(w in q_lower for w in ["time", "date", "trend", "temporal", "month", "year", "day", "trajectory", "over time", "history", "seasonal"]):
            selected_keys.add("temporal")

        # 5. General Insights & Patterns
        if any(w in q_lower for w in ["pattern", "finding", "insight", "takeaway", "overview", "summary", "behavior", "general", "tell me"]):
            selected_keys.add("insights")

        # 6. Numeric Statistics
        if any(w in q_lower for w in ["average", "mean", "median", "sum", "total", "min", "max", "highest", "lowest", "std", "standard deviation", "range"]):
            selected_keys.add("numeric")

        # 7. Categorical Information
        if any(w in q_lower for w in ["category", "breakdown", "segment", "frequent", "frequency", "most common", "top", "rank", "count"]):
            selected_keys.add("categorical")

        # If specific columns were mentioned, inspect their types
        if mentioned_cols:
            for col in mentioned_cols:
                ctype = col_types.get(col, "")
                if ctype == "Numeric":
                    selected_keys.add("numeric")
                elif ctype in ["Categorical", "Boolean"]:
                    selected_keys.add("categorical")
                elif ctype == "Date":
                    selected_keys.add("temporal")

        # If no specific analytical category was triggered, include quality and numeric summary
        if len(selected_keys) == 1:  # Only profile was selected
            selected_keys.update({"quality", "numeric", "insights"})

        # Build output text
        output_chunks = []

        # Always place Verified Ground Truth first if present
        if verified_result and verified_result.get("answer"):
            output_chunks.append(
                f"=== VERIFIED PYTHON GROUND TRUTH (CALCULATED VIA PANDAS) ===\n"
                f"Metric Highlight: {verified_result.get('metric_highlight', 'Verified Fact')}\n"
                f"Calculated Answer: {verified_result.get('answer')}\n"
                f"Engine: Deterministic Python / Pandas Calculation (Authoritative Source of Truth)"
            )

        # Append selected knowledge sections in logical order
        ordered_keys = ["profile", "quality", "numeric", "categorical", "relationships", "temporal", "anomalies", "insights"]
        for k in ordered_keys:
            if k in selected_keys and k in all_sections:
                output_chunks.append(all_sections[k])

        return "\n\n".join(output_chunks)


# Backwards compatibility alias
RAGRetriever = RAGEngine
