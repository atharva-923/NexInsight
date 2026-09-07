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
        all_anomalies: List[Dict[str, Any]] = []
        column_breakdown: Dict[str, Dict[str, Any]] = {}

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

            column_breakdown[col] = {
                "count": len(outlier_series),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "mean": round(mean_val, 2),
                "std": round(std_val, 2)
            }

            for idx, val in outlier_series.items():
                val_float = float(val)
                z_score = abs(val_float - mean_val) / std_val if std_val > 0 else 0.0
                
                # Determine severity
                if z_score >= 3.5 or val_float > (upper_bound + 1.5 * iqr) or val_float < (lower_bound - 1.5 * iqr):
                    severity = "Critical"
                elif z_score >= 2.5:
                    severity = "Moderate"
                else:
                    severity = "Low"

                direction = "above upper threshold" if val_float > upper_bound else "below lower threshold"
                reason = f"Value {val_float:,.2f} is significantly {direction} ({lower_bound:,.2f} to {upper_bound:,.2f})."

                all_anomalies.append({
                    "row_index": int(idx),
                    "column": col,
                    "value": round(val_float, 2),
                    "normal_range": f"{lower_bound:,.2f} - {upper_bound:,.2f}",
                    "deviation": round(abs(val_float - mean_val), 2),
                    "z_score": round(float(z_score), 2),
                    "severity": severity,
                    "reason": reason
                })

        # Sort all anomalies by z_score descending
        all_anomalies.sort(key=lambda x: x["z_score"], reverse=True)

        severity_counts = {
            "Critical": sum(1 for a in all_anomalies if a["severity"] == "Critical"),
            "Moderate": sum(1 for a in all_anomalies if a["severity"] == "Moderate"),
            "Low": sum(1 for a in all_anomalies if a["severity"] == "Low"),
        }

        return {
            "total_anomalies": len(all_anomalies),
            "severity_counts": severity_counts,
            "column_breakdown": column_breakdown,
            "anomalies_list": all_anomalies
        }
