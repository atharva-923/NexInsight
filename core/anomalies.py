"""
NexInsight - Dedicated Anomaly & Outlier Engine
Provides deep anomaly scoring, IQR boundary checks, Z-score calculations,
and structured records for the Anomalies page.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


class AnomalyDetector:
    """Detects and ranks anomalies across all numerical dimensions."""

    def __init__(self, df: pd.DataFrame, column_types: Dict[str, str]):
        self.df = df
        self.column_types = column_types
        self.numeric_cols = [c for c, t in column_types.items() if t == "Numeric" and c in df.columns]

    def get_comprehensive_anomalies(self) -> Dict[str, Any]:
        """
        Scans all numeric columns and extracts flagged anomalies with rich metadata:
        - column
        - row_index
        - value
        - normal_range
        - z_score
        - severity (High, Moderate, Low)
        - reason
        """
        column_breakdown: Dict[str, Dict[str, Any]] = {}
        total_anomalies_count = 0
        total_critical = 0
        total_moderate = 0
        total_low = 0
        candidate_records: List[Dict[str, Any]] = []

        for col in self.numeric_cols:
            series = pd.to_numeric(self.df[col], errors="coerce").dropna()
            if len(series) < 6:
                continue

            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            mean_val = float(series.mean())
            std_val = float(series.std()) if len(series) > 1 else 1.0

            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            outliers_mask = (series < lower_bound) | (series > upper_bound)
            outlier_series = series[outliers_mask]
            n_outliers = len(outlier_series)

            column_breakdown[col] = {
                "count": n_outliers,
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "mean": round(mean_val, 2),
                "std": round(std_val, 2)
            }

            if n_outliers == 0:
                continue

            total_anomalies_count += n_outliers

            # Vectorized z-score and severity calculation
            vals = outlier_series.values
            indices = outlier_series.index
            z_scores = np.abs(vals - mean_val) / std_val if std_val > 0 else np.zeros_like(vals)

            crit_mask = (z_scores >= 3.5) | (vals > (upper_bound + 1.5 * iqr)) | (vals < (lower_bound - 1.5 * iqr))
            mod_mask = (z_scores >= 2.5) & (~crit_mask)
            low_mask = (~crit_mask) & (~mod_mask)

            total_critical += int(crit_mask.sum())
            total_moderate += int(mod_mask.sum())
            total_low += int(low_mask.sum())

            # Collect top anomalies per column for detailed tabular display
            max_col_records = min(n_outliers, 100)
            if n_outliers > max_col_records:
                top_order = np.argsort(z_scores)[-max_col_records:][::-1]
            else:
                top_order = np.argsort(z_scores)[::-1]

            for i in top_order:
                val_float = float(vals[i])
                idx = indices[i]
                z_score = float(z_scores[i])
                severity = "Critical" if crit_mask[i] else ("Moderate" if mod_mask[i] else "Low")
                direction = "above upper threshold" if val_float > upper_bound else "below lower threshold"
                reason = f"Value {val_float:,.2f} is significantly {direction} ({lower_bound:,.2f} to {upper_bound:,.2f})."

                candidate_records.append({
                    "row_index": int(idx),
                    "column": col,
                    "value": round(val_float, 2),
                    "normal_range": f"{lower_bound:,.2f} - {upper_bound:,.2f}",
                    "deviation": round(abs(val_float - mean_val), 2),
                    "z_score": round(z_score, 2),
                    "severity": severity,
                    "reason": reason
                })

        # Sort candidate records by z_score descending
        candidate_records.sort(key=lambda x: x["z_score"], reverse=True)

        return {
            "total_anomalies": total_anomalies_count,
            "severity_counts": {
                "Critical": total_critical,
                "Moderate": total_moderate,
                "Low": total_low
            },
            "column_breakdown": column_breakdown,
            "anomalies_list": candidate_records[:300]
        }
