"""
NexInsight - Factual AI Insight Synthesis Engine
Generates human-readable, executive insights derived strictly from
computed statistical and machine learning results (zero hallucinations).
Supports optional LLM enhancement when an API key is provided.
"""

from typing import Dict, Any, List, Optional
import os


class InsightEngine:
    """Transforms raw statistical calculations into structured executive insights."""

    @staticmethod
    def generate_factual_insights(
        analysis_results: Dict[str, Any],
        health_metrics: Dict[str, Any],
        cleaning_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesizes factual observations based solely on verified metrics.
        Categories:
        - Key Findings
        - Important Patterns & Relationships
        - Anomaly & Risk Alerts
        - Actionable Observations
        - Executive Summary
        """
        key_findings: List[str] = []
        patterns: List[str] = []
        anomalies: List[str] = []
        observations: List[str] = []

        desc_stats = analysis_results.get("descriptive_stats", {})
        cat_stats = analysis_results.get("categorical_analysis", {})
        correlations = analysis_results.get("correlations", {})
        outliers = analysis_results.get("outliers", {})
        clustering = analysis_results.get("clustering")
        trends = analysis_results.get("temporal_trends")

        # 1. Health & Ingestion Finding
        health_score = health_metrics.get("score", 100)
        total_rows = health_metrics.get("total_rows", 0)
        total_cols = health_metrics.get("total_columns", 0)
        dups_removed = cleaning_summary.get("duplicates_removed", 0)
        missing_handled = cleaning_summary.get("missing_values_handled", 0)

        key_findings.append(
            f"The dataset contains {total_rows:,} records across {total_cols} attributes with an overall Health Score of {health_score}/100."
        )
        if dups_removed > 0 or missing_handled > 0:
            key_findings.append(
                f"Data hygiene pipeline sanitized {dups_removed:,} duplicate entries and resolved {missing_handled:,} missing data points."
            )

        # 2. Categorical Dominance
        for col, data in cat_stats.items():
            top_cats = data.get("top_categories", [])
            if top_cats:
                top = top_cats[0]
                patterns.append(
                    f"In '{col}', '{top['category']}' represents the primary segment ({top['percentage']}% of records, {top['count']:,} occurrences)."
                )

        # 3. Numeric Highlights & Skewness
        for col, stats in desc_stats.items():
            mean_v = stats.get("mean", 0)
            median_v = stats.get("median", 0)
            skew_v = stats.get("skewness", 0)
            
            if abs(skew_v) > 1.2:
                direction = "right-skewed (concentrated at lower values with extreme high peaks)" if skew_v > 0 else "left-skewed (concentrated at higher values)"
                patterns.append(
                    f"The distribution for '{col}' is heavily {direction} with an average of {mean_v:,.2f} versus a median of {median_v:,.2f}."
                )

        # 4. Significant Correlations
        if correlations.get("has_correlation"):
            sig_pairs = correlations.get("significant_pairs", [])
            for pair in sig_pairs[:3]:
                if pair["abs_correlation"] >= 0.4:
                    dir_text = "directly co-move" if pair["direction"] == "Positive" else "inversely diverge"
                    patterns.append(
                        f"Significant {pair['strength'].lower()} {pair['direction'].lower()} correlation identified between '{pair['col1']}' and '{pair['col2']}' (r = {pair['correlation']:.2f}); these attributes {dir_text}."
                    )

        # 5. Temporal Trends
        if trends:
            direction = trends.get("overall_direction", "Stable")
            peak = trends.get("peak_period", "N/A")
            metric = trends.get("metric_column", "Metric")
            key_findings.append(
                f"Temporal trajectory for '{metric}' shows an overall {direction.lower()} momentum, with peak volume recorded in {peak}."
            )

        # 6. Anomalies & Outliers
        total_outliers = outliers.get("total_outliers_detected", 0)
        if total_outliers > 0:
            anomalies.append(
                f"Identified {total_outliers:,} potential anomalies across numerical parameters exceeding standard interquartile tolerance."
            )
            top_anoms = outliers.get("top_anomalies", [])
            if top_anoms:
                highest = top_anoms[0]
                anomalies.append(
                    f"Most extreme deviation noted in '{highest['column']}' at row #{highest['row_index']} (value {highest['value']:,.2f}, normal range: {highest['normal_range']})."
                )
        else:
            anomalies.append("No critical statistical anomalies or extreme distribution outliers were flagged.")

        # 7. Cluster Observations
        if clustering and clustering.get("applicable"):
            n_clusters = clustering.get("n_clusters", 0)
            profiles = clustering.get("profiles", [])
            observations.append(
                f"K-Means partitioning uncovered {n_clusters} distinct cohesive behavioral segments within the numerical features."
            )
            for prof in profiles[:2]:
                observations.append(
                    f"Segment '{prof['name']}' represents {prof['percentage']}% of population ({prof['key_traits']})."
                )

        # 8. Actionable Guidance
        if health_score < 80:
            observations.append("Prioritize upstream data collection validation to minimize null values and duplicate submissions.")
        if total_outliers > 0:
            observations.append("Isolate flagged anomaly records for domain audit to prevent statistical distortion in operational planning.")
        if len(observations) == 0:
            observations.append("The dataset demonstrates consistent distributions suitable for predictive modeling and executive reporting.")

        # Concise Executive Summary Paragraph
        summary_lines = [
            f"The analyzed dataset demonstrates a {health_metrics.get('status', 'Good').lower()} data quality profile ({health_score}/100) across {total_rows:,} records.",
            patterns[0] if patterns else f"Analyzed across {total_cols} dimensions without structural anomalies.",
            anomalies[0] if anomalies else "Data exhibits smooth parameter distributions."
        ]
        executive_summary = " ".join(summary_lines)

        return {
            "executive_summary": executive_summary,
            "key_findings": key_findings[:5],
            "important_patterns": patterns[:6],
            "anomalies": anomalies[:4],
            "observations": observations[:5]
        }

    @classmethod
    def synthesize_with_llm(
        cls,
        analysis_data: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Optional LLM connector (e.g. Gemini / OpenAI) if user configures an API key.
        Uses exact calculated facts as context to draft an executive memo.
        """
        if not api_key:
            return None

        # If Gemini key or environment key is present
        try:
            import requests
            prompt = (
                "You are an elite Senior Data Analyst at McKinsey. "
                "Synthesize the following strictly computed dataset analysis into an executive summary memo for leadership. "
                "Do NOT invent any numbers. Use only the provided statistics.\n\n"
                f"ANALYSIS DATA:\n{str(analysis_data)[:3500]}\n\n"
                "Provide: Executive Brief, Strategic Takeaways, and Risk Warnings in clean markdown."
            )
            # Example Gemini REST call
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            res = requests.post(url, json=payload, timeout=12)
            if res.status_code == 200:
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return None
        return None
