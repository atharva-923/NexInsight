"""
NexInsight - Natural Language 'Ask Your Data' Engine
Interprets plain-English analytical queries and extracts precise,
data-backed answers with supporting metrics, charts, and data slices.
"""

import re
import difflib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from core.visualizer import Visualizer


class DataQAEngine:
    """Interprets user queries and computes factual responses against live dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        column_types: Dict[str, str],
        analysis_results: Dict[str, Any],
        health_metrics: Dict[str, Any]
    ):
        self.df = df
        self.column_types = column_types
        self.analysis = analysis_results
        self.health = health_metrics
        self.numeric_cols = [c for c, t in column_types.items() if t == "Numeric" and c in df.columns]
        self.categorical_cols = [c for c, t in column_types.items() if t in ["Categorical", "Boolean"] and c in df.columns]
        self.date_cols = [c for c, t in column_types.items() if t == "Date" and c in df.columns]

    def _find_matching_column(self, query: str) -> Optional[str]:
        """Finds column name mentioned in query via substring or fuzzy match."""
        q_clean = query.lower()
        # Direct exact or substring check
        for col in self.df.columns:
            if col.lower() in q_clean:
                return col
        
        # Word-level fuzzy match
        words = re.findall(r"\w+", q_clean)
        best_match = None
        best_score = 0.0
        for w in words:
            if len(w) <= 2:
                continue
            matches = difflib.get_close_matches(w, [c.lower() for c in self.df.columns], n=1, cutoff=0.7)
            if matches:
                for col in self.df.columns:
                    if col.lower() == matches[0]:
                        return col
        return None

    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Routes and processes query using semantic pattern matching and live calculations.
        Returns:
        {
            "query": str,
            "answer": str,
            "metric_highlight": Optional[str],
            "figure": Optional[go.Figure],
            "data_slice": Optional[pd.DataFrame]
        }
        """
        q = query.lower().strip()
        matched_col = self._find_matching_column(q)

        # 1. Anomalies / Unusual / Outliers
        if any(w in q for w in ["unusual", "anomaly", "anomalies", "outlier", "outliers", "flagged", "strange"]):
            outlier_data = self.analysis.get("outliers", {})
            total_anoms = outlier_data.get("total_outliers_detected", 0)
            top_records = outlier_data.get("top_anomalies", [])
            
            if total_anoms == 0:
                return {
                    "query": query,
                    "answer": "No critical statistical anomalies or extreme outliers were detected in this dataset.",
                    "metric_highlight": "0 Anomalies Detected",
                    "figure": None,
                    "data_slice": None
                }
            
            highest = top_records[0] if top_records else {}
            ans = (
                f"Yes, NexInsight detected **{total_anoms:,} potential anomalies** across numerical parameters. "
                f"The most extreme occurrence is in attribute **'{highest.get('column')}'** at row #{highest.get('row_index')} "
                f"with value **{highest.get('value'):,}** (expected normal range: {highest.get('normal_range')})."
            )
            
            fig = None
            if highest.get("column") and highest.get("column") in self.df.columns:
                fig = Visualizer.create_anomaly_scatter(self.df, highest["column"], top_records)

            df_slice = pd.DataFrame(top_records[:10]) if top_records else None
            return {
                "query": query,
                "answer": ans,
                "metric_highlight": f"{total_anoms} Anomalies Identified",
                "figure": fig,
                "data_slice": df_slice
            }

        # 2. Strongest correlations / Relationships
        if any(w in q for w in ["correlation", "correlations", "relationship", "correlated", "co-move", "related"]):
            corr_data = self.analysis.get("correlations", {})
            if not corr_data.get("has_correlation") or not corr_data.get("significant_pairs"):
                return {
                    "query": query,
                    "answer": "There are not enough numerical dimensions in this dataset to establish statistically significant correlations.",
                    "metric_highlight": "No Correlation Data",
                    "figure": None,
                    "data_slice": None
                }
            
            top_pair = corr_data["significant_pairs"][0]
            ans = (
                f"The strongest statistical relationship is between **'{top_pair['col1']}'** and **'{top_pair['col2']}'** "
                f"with a **{top_pair['strength'].lower()} {top_pair['direction'].lower()} correlation** of **r = {top_pair['correlation']:.2f}**. "
                f"When '{top_pair['col1']}' changes, '{top_pair['col2']}' tends to {'increase' if top_pair['direction'] == 'Positive' else 'decrease'} proportionally."
            )
            fig = Visualizer.create_scatter_plot(self.df, top_pair["col1"], top_pair["col2"])
            return {
                "query": query,
                "answer": ans,
                "metric_highlight": f"r = {top_pair['correlation']:.2f} ({top_pair['strength']})",
                "figure": fig,
                "data_slice": pd.DataFrame(corr_data["significant_pairs"][:6])
            }

        # 3. Temporal Peaks / Months / Trend
        if any(w in q for w in ["month", "time", "trend", "peak", "temporal", "period", "grow", "growth"]):
            trends = self.analysis.get("temporal_trends")
            if trends:
                ans = (
                    f"Based on chronological analysis of **'{trends['metric_column']}'** across **'{trends['date_column']}'**: "
                    f"The overall trajectory is **{trends['overall_direction']}** ({trends['trend_percentage']:+0.1f}% change). "
                    f"The highest activity was recorded in **{trends['peak_period']}**, while the lowest was in **{trends['lowest_period']}**."
                )
                fig = Visualizer.create_time_series_chart(self.df, trends["date_column"], trends["metric_column"])
                return {
                    "query": query,
                    "answer": ans,
                    "metric_highlight": f"Peak: {trends['peak_period']}",
                    "figure": fig,
                    "data_slice": None
                }

        # 4. Highest / Top / Best / Maximum
        if any(w in q for w in ["highest", "top", "best", "max", "maximum", "peak"]):
            # If a specific numeric column was asked
            if matched_col and matched_col in self.numeric_cols:
                series = pd.to_numeric(self.df[matched_col], errors="coerce").dropna()
                max_val = series.max()
                max_row = self.df.loc[series.idxmax()]
                ans = f"The maximum value for **'{matched_col}'** is **{max_val:,.2f}**."
                fig = Visualizer.create_histogram(self.df, matched_col)
                return {
                    "query": query,
                    "answer": ans,
                    "metric_highlight": f"Max {matched_col}: {max_val:,.2f}",
                    "figure": fig,
                    "data_slice": pd.DataFrame([max_row])
                }

            # If highest performing category is asked
            if self.categorical_cols and self.numeric_cols:
                cat_col = self.categorical_cols[0]
                num_col = self.numeric_cols[0]
                grouped = self.df.groupby(cat_col)[num_col].agg(["sum", "mean", "count"]).reset_index()
                grouped = grouped.sort_values(by="sum", ascending=False)
                top_row = grouped.iloc[0]
                
                ans = (
                    f"The highest-performing category in **'{cat_col}'** (measured by total **'{num_col}'**) "
                    f"is **'{top_row[cat_col]}'** with a total of **{top_row['sum']:,.2f}** "
                    f"(average: {top_row['mean']:,.2f} across {int(top_row['count'])} records)."
                )
                fig = Visualizer.create_category_bar_chart(self.df, cat_col, num_col, agg="sum")
                return {
                    "query": query,
                    "answer": ans,
                    "metric_highlight": f"Top: {top_row[cat_col]} ({top_row['sum']:,.2f})",
                    "figure": fig,
                    "data_slice": grouped.head(6)
                }

        # 5. Lowest / Minimum / Smallest
        if any(w in q for w in ["lowest", "bottom", "worst", "min", "minimum", "smallest"]):
            if matched_col and matched_col in self.numeric_cols:
                series = pd.to_numeric(self.df[matched_col], errors="coerce").dropna()
                min_val = series.min()
                min_row = self.df.loc[series.idxmin()]
                ans = f"The minimum value for **'{matched_col}'** is **{min_val:,.2f}**."
                fig = Visualizer.create_histogram(self.df, matched_col)
                return {
                    "query": query,
                    "answer": ans,
                    "metric_highlight": f"Min {matched_col}: {min_val:,.2f}",
                    "figure": fig,
                    "data_slice": pd.DataFrame([min_row])
                }

        # 6. Average / Mean / Median
        if any(w in q for w in ["average", "mean", "median", "typical"]):
            target = matched_col if matched_col in self.numeric_cols else (self.numeric_cols[0] if self.numeric_cols else None)
            if target:
                series = pd.to_numeric(self.df[target], errors="coerce").dropna()
                mean_v = series.mean()
                median_v = series.median()
                std_v = series.std()
                ans = (
                    f"For parameter **'{target}'**, the average (mean) is **{mean_v:,.2f}** "
                    f"with a median of **{median_v:,.2f}** and standard deviation of **{std_v:,.2f}**."
                )
                fig = Visualizer.create_histogram(self.df, target)
                return {
                    "query": query,
                    "answer": ans,
                    "metric_highlight": f"Mean {target}: {mean_v:,.2f}",
                    "figure": fig,
                    "data_slice": None
                }

        # 7. Important Patterns / Findings / Summary
        if any(w in q for w in ["pattern", "patterns", "finding", "findings", "insight", "insights", "summary", "overview"]):
            cat_info = []
            for c in self.categorical_cols[:2]:
                top_v = self.df[c].value_counts().index[0]
                pct = (self.df[c].value_counts().iloc[0] / len(self.df)) * 100
                cat_info.append(f"'{c}' is dominated by '{top_v}' ({pct:.1f}%)")
            
            corr_info = ""
            corr_data = self.analysis.get("correlations", {})
            if corr_data.get("has_correlation") and corr_data.get("significant_pairs"):
                top_p = corr_data["significant_pairs"][0]
                corr_info = f" Strongest interaction is between '{top_p['col1']}' and '{top_p['col2']}' (r={top_p['correlation']:.2f})."

            ans = (
                f"**Key patterns in this dataset:**\n\n"
                f"- Evaluated **{len(self.df):,} rows** across **{len(self.df.columns)} columns** with health score **{self.health.get('score', 100)}/100**.\n"
                f"- {'; '.join(cat_info) if cat_info else 'Balanced categorical distribution.'}\n"
                f"- {corr_info}\n"
                f"- Total of **{self.analysis.get('outliers', {}).get('total_outliers_detected', 0)}** statistical anomalies identified."
            )
            fig = Visualizer.auto_generate_dashboard_charts(self.df, self.column_types)[0] if self.numeric_cols else None
            return {
                "query": query,
                "answer": ans,
                "metric_highlight": f"Dataset Health: {self.health.get('score', 100)}/100",
                "figure": fig,
                "data_slice": None
            }

        # 8. Generic Column Inquiry or Fallback
        if matched_col:
            col_type = self.column_types.get(matched_col, "Unknown")
            if col_type == "Numeric":
                series = pd.to_numeric(self.df[matched_col], errors="coerce").dropna()
                ans = (
                    f"Attribute **'{matched_col}'** is a numerical feature with {len(series):,} valid values. "
                    f"Mean: **{series.mean():,.2f}**, Median: **{series.median():,.2f}**, Min: **{series.min():,.2f}**, Max: **{series.max():,.2f}**."
                )
                fig = Visualizer.create_histogram(self.df, matched_col)
                return {
                    "query": query,
                    "answer": ans,
                    "metric_highlight": f"Mean: {series.mean():,.2f}",
                    "figure": fig,
                    "data_slice": self.df[[matched_col]].describe().reset_index()
                }
            elif col_type in ["Categorical", "Boolean"]:
                vc = self.df[matched_col].value_counts().head(5)
                top_item = vc.index[0]
                top_cnt = vc.iloc[0]
                ans = (
                    f"Attribute **'{matched_col}'** contains {self.df[matched_col].nunique()} distinct categories. "
                    f"Most frequent value is **'{top_item}'** with {top_cnt:,} records ({top_cnt / len(self.df) * 100:.1f}%)."
                )
                fig = Visualizer.create_donut_chart(self.df, matched_col)
                return {
                    "query": query,
                    "answer": ans,
                    "metric_highlight": f"Mode: {top_item}",
                    "figure": fig,
                    "data_slice": vc.reset_index()
                }

        # Default fallback response
        fallback_ans = (
            f"Query analyzed against {len(self.df):,} rows and {len(self.df.columns)} attributes. "
            f"Detected dimensions: {', '.join(self.df.columns[:6])}. "
            "Try asking: 'What is the highest-performing category?', 'Are there unusual records?', 'What are the strongest correlations?', or ask about any specific column."
        )
        return {
            "query": query,
            "answer": fallback_ans,
            "metric_highlight": f"{len(self.df.columns)} Columns Active",
            "figure": None,
            "data_slice": None
        }
