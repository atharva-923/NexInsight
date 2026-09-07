"""
NexInsight - Natural Language 'Ask Your Data' Engine
Interprets plain-English analytical queries, calculates ground-truth statistics via Pandas,
retrieves targeted tabular context via RAG, and synthesizes natural-language answers
via xAI Grok API with seamless local fallback.
"""

import re
import difflib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import plotly.express as px
import plotly.graph_objects as go
from core.visualizer import Visualizer
from core.llm_client import LLMClient, GrokClient
from core.rag_engine import RAGEngine, RAGRetriever


class DataQAEngine:
    """Hybrid Data QA engine combining deterministic Pandas truth with Grok natural reasoning."""

    def __init__(
        self,
        df: pd.DataFrame,
        column_types: Dict[str, str],
        analysis_results: Dict[str, Any],
        health_metrics: Dict[str, Any],
        dataset_record: Optional[Dict[str, Any]] = None
    ):
        self.df = df
        self.column_types = column_types
        self.analysis = analysis_results
        self.health = health_metrics
        self.numeric_cols = [c for c, t in column_types.items() if t == "Numeric" and c in df.columns]
        self.categorical_cols = [c for c, t in column_types.items() if t in ["Categorical", "Boolean"] and c in df.columns]
        self.date_cols = [c for c, t in column_types.items() if t == "Date" and c in df.columns]

        # Construct or use comprehensive dataset record for RAG retrieval
        if dataset_record:
            self.dataset_record = dataset_record
        else:
            self.dataset_record = {
                "filename": "active_dataset.csv",
                "raw_df": df,
                "cleaned_df": df,
                "cleaned_row_count": len(df),
                "original_row_count": len(df),
                "column_count": len(df.columns),
                "column_types": column_types,
                "missing_value_count": health_metrics.get("total_missing", 0),
                "duplicate_count": 0,
                "health_metrics": health_metrics,
                "cleaning_summary": {},
                "analysis_results": analysis_results,
                "anomalies_data": analysis_results.get("outliers", {}),
                "ai_insights": {}
            }

    def _find_matching_column(self, query: str) -> Optional[str]:
        """Finds column name mentioned in query via word boundary or exact match."""
        q_clean = query.lower()
        
        # 1. Direct exact column name check with word boundaries
        for col in self.df.columns:
            c_low = col.lower()
            if re.search(r"\b" + re.escape(c_low) + r"\b", q_clean):
                return col

        # 2. Check compound column tokens (e.g. 'revenue' in 'Total_Revenue' or 'cpu' in 'CPU_Load_Pct')
        for col in self.df.columns:
            c_low = col.lower()
            tokens = [t for t in re.split(r"[_\s\-]+", c_low) if len(t) >= 4]
            for tok in tokens:
                if re.search(r"\b" + re.escape(tok) + r"\b", q_clean):
                    return col
        
        # 3. Word-level fuzzy match only for words >= 5 chars with high cutoff
        words = re.findall(r"\b[a-zA-Z]{5,}\b", q_clean)
        for w in words:
            matches = difflib.get_close_matches(w, [c.lower() for c in self.df.columns], n=1, cutoff=0.85)
            if matches:
                for col in self.df.columns:
                    if col.lower() == matches[0]:
                        return col
        return None

    def compute_deterministic_answer(self, query: str) -> Dict[str, Any]:
        """
        Calculates ground-truth numerical and statistical answers using Pandas.
        Zero hallucinations: every single metric is verified against the live dataframe.
        """
        if self.df is None or self.df.empty:
            return {
                "query": query,
                "answer": "The dataset is empty (0 records). No statistical calculations can be performed.",
                "metric_highlight": "0 Records Available",
                "figure": None,
                "data_slice": None
            }

        q = query.lower().strip()
        matched_col = self._find_matching_column(q)

        # 0. Greetings / Help
        if q in ["hi", "hello", "hey", "help", "who are you", "what can you do", "hi!", "hello!"]:
            filename = self.dataset_record.get("filename", "the active dataset")
            ans = (
                f"Hello! I am **NexInsight AI Analyst**, your autonomous data intelligence assistant for **'{filename}'** "
                f"({len(self.df):,} rows, {len(self.df.columns)} attributes). "
                "I combine verified Python calculations with natural-language insights. "
                "You can ask me questions like: *'Which region has the highest sales?'*, *'What columns have missing values?'*, "
                "*'Are there unusual observations?'*, or *'What are the strongest correlations?'*."
            )
            return {
                "query": query,
                "answer": ans,
                "metric_highlight": f"Analyst Ready ({len(self.df):,} Rows)",
                "figure": None,
                "data_slice": None
            }

        # 1. Row count / Size
        if any(w in q for w in ["how many rows", "row count", "record count", "total rows", "number of rows", "how many records", "how big", "dataset size"]):
            ans = f"The dataset contains **{len(self.df):,} rows** and **{len(self.df.columns)} columns** with a Health Score of **{self.health.get('score', 100)}/100**."
            return {
                "query": query,
                "answer": ans,
                "metric_highlight": f"{len(self.df):,} Rows · {len(self.df.columns)} Columns",
                "figure": None,
                "data_slice": None
            }

        # 2. Missing values / Duplicates / Data Quality
        if any(w in q for w in ["missing", "null", "nan", "duplicate", "duplicates", "data quality", "hygiene", "health score", "completeness"]):
            raw_df = self.dataset_record.get("raw_df")
            eval_df = raw_df if raw_df is not None else self.df
            missing_series = eval_df.isna().sum()
            cols_with_missing = missing_series[missing_series > 0]
            total_missing = int(missing_series.sum())
            clean_summary = self.dataset_record.get("cleaning_summary", {})
            dup_count = clean_summary.get("duplicates_removed", clean_summary.get("duplicates_detected", 0))
            score = self.health.get("score", 100)

            if any(w in q for w in ["duplicate", "duplicates"]):
                ans = f"NexInsight detected and resolved **{dup_count:,} duplicate rows** in this dataset (Uniqueness index: {self.health.get('uniqueness_pct', 100.0):.1f}%)."
                highlight = f"{dup_count:,} Duplicates"
            elif cols_with_missing.empty:
                ans = (
                    f"**Zero missing values** were detected across all {len(eval_df.columns)} columns in this dataset. "
                    f"Data completeness is **100.0%** and the overall Health Score is **{score}/100**."
                )
                highlight = "0 Missing Values (100% Complete)"
            else:
                breakdown = [f"**'{c}'** ({cnt:,} missing, {cnt/len(eval_df)*100:.1f}%)" for c, cnt in cols_with_missing.items()]
                ans = (
                    f"The following columns contain missing values in the raw dataset: {', '.join(breakdown)}. "
                    f"In total, there were **{total_missing:,} missing values** (Completeness index: {self.health.get('completeness_pct', 100.0):.1f}%, Health Score: {score}/100)."
                )
                highlight = f"{total_missing:,} Missing Values ({len(cols_with_missing)} Columns)"

            missing_df = pd.DataFrame({"Column": missing_series.index, "Missing Count": missing_series.values})
            return {
                "query": query,
                "answer": ans,
                "metric_highlight": highlight,
                "figure": None,
                "data_slice": missing_df[missing_df["Missing Count"] > 0] if not cols_with_missing.empty else None
            }

        # 3. Anomalies / Unusual / Outliers
        if any(w in q for w in ["unusual", "anomaly", "anomalies", "outlier", "outliers", "flagged", "strange", "abnormal"]):
            outlier_data = self.analysis.get("outliers", {})
            anoms_data = self.dataset_record.get("anomalies_data", {})
            total_anoms = anoms_data.get("total_anomalies", outlier_data.get("total_outliers_detected", 0))
            top_records = anoms_data.get("anomalies_list", outlier_data.get("top_anomalies", []))
            sev_counts = anoms_data.get("severity_counts", {})
            
            if total_anoms == 0:
                return {
                    "query": query,
                    "answer": "No critical statistical anomalies or extreme outliers were detected in this dataset (0 anomalies identified).",
                    "metric_highlight": "0 Anomalies Detected",
                    "figure": None,
                    "data_slice": None
                }
            
            highest = top_records[0] if top_records else {}
            ans = (
                f"NexInsight detected **{total_anoms:,} potential anomalies** across numerical parameters "
                f"({sev_counts.get('Critical', 0)} critical, {sev_counts.get('Moderate', 0)} moderate). "
                f"The most extreme occurrence is in **'{highest.get('column')}'** at row #{highest.get('row_index')} "
                f"with value **{highest.get('value'):,}** (normal range: {highest.get('normal_range')})."
            )
            
            fig = None
            if highest.get("column") and highest.get("column") in self.df.columns:
                fig = Visualizer.create_anomaly_scatter(self.df, highest["column"], top_records)

            df_slice = pd.DataFrame(top_records[:10]) if top_records else None
            return {
                "query": query,
                "answer": ans,
                "metric_highlight": f"{total_anoms:,} Anomalies Identified",
                "figure": fig,
                "data_slice": df_slice
            }

        # 3. Strongest correlations / Relationships
        if any(w in q for w in ["correlation", "correlations", "relationship", "correlated", "co-move", "related", "dependency", "association"]):
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

        # 4. Temporal Peaks / Months / Trends
        if any(w in q for w in ["month", "time", "trend", "peak", "temporal", "period", "grow", "growth", "seasonality"]):
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

        # 5. Highest / Top / Best / Maximum
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
            if self.categorical_cols and self.numeric_cols and any(w in q for w in ["category", "segment", "perform", "product", "sales", "revenue", "tier", "region", "lead", "rank", "grouped"]):
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

        # 6. Lowest / Minimum / Smallest
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

        # 7. Sum / Total
        if any(w in q for w in ["sum", "total of", "sum of", "total amount", "cumulative"]):
            target = matched_col if matched_col in self.numeric_cols else (self.numeric_cols[0] if self.numeric_cols else None)
            if target:
                series = pd.to_numeric(self.df[target], errors="coerce").dropna()
                tot_val = series.sum()
                ans = f"The total (sum) of **'{target}'** across {len(series):,} valid records is **{tot_val:,.2f}**."
                fig = Visualizer.create_histogram(self.df, target)
                return {
                    "query": query,
                    "answer": ans,
                    "metric_highlight": f"Sum {target}: {tot_val:,.2f}",
                    "figure": fig,
                    "data_slice": None
                }

        # 8. Average / Mean / Median
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

        # 8. Most frequent category / Common
        if any(w in q for w in ["most frequent", "most common", "occurs most", "dominant category", "mode", "frequency"]):
            target_cat = matched_col if matched_col in self.categorical_cols else (self.categorical_cols[0] if self.categorical_cols else None)
            if target_cat:
                vc = self.df[target_cat].value_counts()
                mode_val = vc.index[0]
                mode_cnt = vc.iloc[0]
                mode_pct = (mode_cnt / len(self.df)) * 100
                ans = (
                    f"In column **'{target_cat}'**, the most frequent category is **'{mode_val}'**, "
                    f"occurring **{mode_cnt:,} times** ({mode_pct:.1f}% of records)."
                )
                fig = Visualizer.create_donut_chart(self.df, target_cat)
                return {
                    "query": query,
                    "answer": ans,
                    "metric_highlight": f"Mode: {mode_val} ({mode_pct:.1f}%)",
                    "figure": fig,
                    "data_slice": vc.head(6).reset_index()
                }

        # 9. Important Patterns / Findings / Summary
        if any(w in q for w in ["pattern", "patterns", "finding", "findings", "insight", "insights", "summary", "overview", "what stands out"]):
            cat_info = []
            for c in self.categorical_cols[:2]:
                vc = self.df[c].value_counts()
                top_v = vc.index[0]
                pct = (vc.iloc[0] / len(self.df)) * 100
                cat_info.append(f"'{c}' is dominated by '{top_v}' ({pct:.1f}%)")
            
            corr_info = ""
            corr_data = self.analysis.get("correlations", {})
            if corr_data.get("has_correlation") and corr_data.get("significant_pairs"):
                top_p = corr_data["significant_pairs"][0]
                corr_info = f" Strongest interaction is between '{top_p['col1']}' and '{top_p['col2']}' (r={top_p['correlation']:.2f})."

            anoms_data = self.dataset_record.get("anomalies_data", {})
            total_anoms = anoms_data.get("total_anomalies", self.analysis.get("outliers", {}).get("total_outliers_detected", 0))

            ans = (
                f"**Key patterns in this dataset:**\n\n"
                f"- Evaluated **{len(self.df):,} rows** across **{len(self.df.columns)} columns** with health score **{self.health.get('score', 100)}/100**.\n"
                f"- {'; '.join(cat_info) if cat_info else 'Balanced categorical distribution.'}\n"
                f"- {corr_info}\n"
                f"- Total of **{total_anoms:,}** statistical anomalies identified."
            )
            fig = Visualizer.auto_generate_dashboard_charts(self.df, self.column_types)[0] if self.numeric_cols else None
            return {
                "query": query,
                "answer": ans,
                "metric_highlight": f"Health: {self.health.get('score', 100)}/100",
                "figure": fig,
                "data_slice": None
            }

        # 10. Column-specific query
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
            f"Active dimensions: {', '.join(self.df.columns[:6])}. "
            "Try asking: 'What is the average of the numeric parameters?', 'What are the strongest correlations?', 'How many anomalies were detected?', or 'What are the main patterns?'."
        )
        return {
            "query": query,
            "answer": fallback_ans,
            "metric_highlight": f"{len(self.df.columns)} Attributes Active",
            "figure": None,
            "data_slice": None
        }

    def answer_query(
        self,
        query: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Orchestrates hybrid QA flow:
        1. Understand user's question and calculate exact deterministic Python ground-truth answer if possible.
        2. Retrieve compact, relevant structured RAG context strictly scoped to active dataset.
        3. If Groq API is configured, prompt Groq for natural-language synthesis of verified results.
        4. If Groq is unavailable, fall back seamlessly to local deterministic engine without crashing.
        """
        # 1. Deterministic Python calculation
        deterministic_res = self.compute_deterministic_answer(query)
        filename = self.dataset_record.get("filename", "active_dataset.csv")

        # 2. Check if Groq API is configured
        if not LLMClient.is_configured(api_key):
            # Local fallback mode
            return {
                "query": query,
                "answer": deterministic_res["answer"],
                "metric_highlight": deterministic_res.get("metric_highlight"),
                "figure": deterministic_res.get("figure"),
                "data_slice": deterministic_res.get("data_slice"),
                "engine": "local",
                "model": "NexInsight Deterministic Engine",
                "fallback_message": "Groq AI Analyst is not configured. Using NexInsight's local analytical engine.",
                "error": None
            }

        # 3. Retrieve compact, grounded RAG context from RAGEngine
        grounded_context = RAGEngine.retrieve_relevant_context(
            query,
            self.dataset_record,
            verified_result=deterministic_res
        )
        system_prompt = RAGEngine.get_system_prompt(filename)
        user_prompt = (
            f"DATASET CONTEXT:\n{grounded_context}\n\n"
            f"USER QUESTION: {query}\n\n"
            "Please provide a natural, concise explanation (2-3 sentences) of this verified calculation. "
            "Do not invent any numbers. Always ground your explanation in the verified figures."
        )

        # 4. Invoke Groq API
        groq_resp = LLMClient.generate_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            conversation_history=conversation_history,
            explicit_key=api_key,
            explicit_model=model,
            temperature=0.2,
            timeout=18
        )

        if groq_resp["success"]:
            return {
                "query": query,
                "answer": groq_resp["content"],
                "metric_highlight": deterministic_res.get("metric_highlight"),
                "figure": deterministic_res.get("figure"),
                "data_slice": deterministic_res.get("data_slice"),
                "engine": "groq",
                "model": groq_resp["model"],
                "fallback_message": None,
                "error": None
            }
        else:
            # Fallback on Groq failure (timeout, network error, rate limit, etc.)
            return {
                "query": query,
                "answer": deterministic_res["answer"],
                "metric_highlight": deterministic_res.get("metric_highlight"),
                "figure": deterministic_res.get("figure"),
                "data_slice": deterministic_res.get("data_slice"),
                "engine": "fallback",
                "model": "NexInsight Deterministic Engine (Fallback)",
                "fallback_message": f"Groq AI Analyst is unavailable ({groq_resp.get('error', 'API error')}). Deterministic dataset analysis is still available.",
                "error": groq_resp.get("error")
            }
