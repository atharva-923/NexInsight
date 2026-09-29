"""
NexInsight - Core Statistical & ML Analyzer
Calculates descriptive statistics, categorical distributions, correlation matrices,
outlier detection (IQR & Z-score), time-series trends, and adaptive K-Means clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class DataAnalyzer:
    """Performs comprehensive automatic statistical and machine learning analysis."""

    def __init__(self, df: pd.DataFrame, column_types: Dict[str, str]):
        self.df = df
        self.column_types = column_types
        self.numeric_cols = [c for c, t in column_types.items() if t == "Numeric" and c in df.columns]
        self.categorical_cols = [c for c, t in column_types.items() if t in ["Categorical", "Boolean"] and c in df.columns]
        self.date_cols = [c for c, t in column_types.items() if t == "Date" and c in df.columns]
        self.text_cols = [c for c, t in column_types.items() if t == "Text" and c in df.columns]

    def compute_descriptive_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Computes descriptive statistics for all numeric columns:
        Count, Mean, Median, Min, Max, Std, IQR, Skewness, Kurtosis.
        """
        stats_dict = {}
        for col in self.numeric_cols:
            series = pd.to_numeric(self.df[col], errors="coerce").dropna()
            if len(series) == 0:
                continue

            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            mean_val = float(series.mean())
            median_val = float(series.median())
            std_val = float(series.std()) if len(series) > 1 else 0.0
            min_val = float(series.min())
            max_val = float(series.max())
            skew_val = float(series.skew()) if len(series) > 2 else 0.0
            kurt_val = float(series.kurt()) if len(series) > 3 else 0.0

            stats_dict[col] = {
                "count": int(len(series)),
                "mean": round(mean_val, 2),
                "median": round(median_val, 2),
                "std": round(std_val, 2),
                "min": round(min_val, 2),
                "max": round(max_val, 2),
                "q25": round(q25, 2),
                "q75": round(q75, 2),
                "iqr": round(iqr, 2),
                "skewness": round(skew_val, 2),
                "kurtosis": round(kurt_val, 2),
                "range": round(max_val - min_val, 2)
            }
        return stats_dict

    def compute_categorical_analysis(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyzes categorical distributions:
        Unique values, mode, top 5 categories with frequencies & percentages.
        """
        cat_dict = {}
        for col in self.categorical_cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue

            val_counts = series.value_counts()
            total = len(series)
            top_cats = []
            for val, count in val_counts.head(5).items():
                pct = round((count / total) * 100, 1)
                top_cats.append({"category": str(val), "count": int(count), "percentage": pct})

            mode_val = str(val_counts.index[0]) if len(val_counts) > 0 else "N/A"
            mode_pct = round((val_counts.iloc[0] / total) * 100, 1) if len(val_counts) > 0 else 0.0

            cat_dict[col] = {
                "unique_count": int(series.nunique()),
                "total_records": int(total),
                "mode": mode_val,
                "mode_percentage": mode_pct,
                "top_categories": top_cats,
                "distribution": {str(k): int(v) for k, v in val_counts.head(10).items()}
            }
        return cat_dict

    def compute_correlations(self) -> Dict[str, Any]:
        """
        Computes Pearson correlation matrix and extracts significant relationships.
        """
        if len(self.numeric_cols) < 2:
            return {"has_correlation": False, "matrix": {}, "significant_pairs": []}

        # Filter numeric df
        num_df = self.df[self.numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(num_df) < 5:
            return {"has_correlation": False, "matrix": {}, "significant_pairs": []}

        corr_matrix = num_df.corr(method="pearson").round(3)
        
        # Extract pairs
        pairs = []
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1, col2 = cols[i], cols[j]
                val = corr_matrix.loc[col1, col2]
                if pd.notna(val):
                    strength = "Weak"
                    if abs(val) >= 0.7:
                        strength = "Very Strong"
                    elif abs(val) >= 0.5:
                        strength = "Moderate"
                    elif abs(val) >= 0.3:
                        strength = "Modest"

                    pairs.append({
                        "col1": col1,
                        "col2": col2,
                        "correlation": float(val),
                        "abs_correlation": abs(float(val)),
                        "direction": "Positive" if val > 0 else "Negative",
                        "strength": strength
                    })

        # Sort by absolute correlation
        pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

        return {
            "has_correlation": True,
            "columns": cols,
            "matrix": corr_matrix.to_dict(),
            "matrix_df": corr_matrix,
            "significant_pairs": pairs
        }

    def detect_outliers(self) -> Dict[str, Any]:
        """
        Detects anomalies and outliers per numeric column using IQR and Z-score methods.
        """
        outlier_summary = {}
        total_outliers = 0
        detailed_records = []

        for col in self.numeric_cols:
            series = pd.to_numeric(self.df[col], errors="coerce").dropna()
            if len(series) < 8:
                continue

            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            mean = series.mean()
            std = series.std()
            z_threshold = 3.0

            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_indices = series[outlier_mask].index

            col_outliers = len(outlier_indices)
            total_outliers += col_outliers

            outlier_summary[col] = {
                "outlier_count": col_outliers,
                "outlier_percentage": round((col_outliers / len(series)) * 100, 2),
                "lower_bound": round(float(lower_bound), 2),
                "upper_bound": round(float(upper_bound), 2),
                "normal_range": f"{lower_bound:.2f} to {upper_bound:.2f}"
            }

            # Collect top 5 most severe outlier records for this column
            for idx in outlier_indices[:5]:
                val = float(series.loc[idx])
                z_score = abs(val - mean) / std if std > 0 else 0
                severity = "High" if z_score >= 3.5 or (val > upper_bound + iqr) else "Moderate"
                
                detailed_records.append({
                    "row_index": int(idx),
                    "column": col,
                    "value": round(val, 2),
                    "normal_range": f"{lower_bound:.2f} - {upper_bound:.2f}",
                    "z_score": round(float(z_score), 2),
                    "severity": severity,
                    "reason": f"Value {val:.2f} is outside 1.5x IQR threshold ({lower_bound:.2f} to {upper_bound:.2f})"
                })

        # Sort detailed records by z-score descending
        detailed_records.sort(key=lambda x: x["z_score"], reverse=True)

        return {
            "total_outliers_detected": total_outliers,
            "column_outliers": outlier_summary,
            "top_anomalies": detailed_records[:20]
        }

    def compute_clustering(self) -> Optional[Dict[str, Any]]:
        """
        Executes adaptive K-Means clustering if >= 2 numeric features and >= 15 records exist.
        Scales features and identifies cluster profiles.
        """
        if len(self.numeric_cols) < 2 or len(self.df) < 15:
            return None

        # Prepare clean numeric data
        features_df = self.df[self.numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(features_df) < 15:
            return None

        try:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_df)
            n_samples = len(scaled_features)

            # Determine best k between 2 and min(5, n_samples - 1)
            max_k = min(5, n_samples // 5)
            if max_k < 2:
                max_k = 2

            best_k = 3 if max_k >= 3 else 2
            best_score = -1.0

            # Representative sample for silhouette score to eliminate O(N^2) bottleneck on large datasets
            eval_sample_size = min(n_samples, 2000)
            rng = np.random.RandomState(42)
            eval_idx = rng.choice(n_samples, size=eval_sample_size, replace=False) if n_samples > eval_sample_size else None
            eval_features = scaled_features[eval_idx] if eval_idx is not None else scaled_features

            for k in range(2, max_k + 1):
                if n_samples > 25000:
                    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=3)
                else:
                    km = KMeans(n_clusters=k, random_state=42, n_init=5)
                
                # Fit model
                if n_samples > 25000 and eval_idx is not None:
                    km.fit(eval_features)
                    eval_labels = km.predict(eval_features)
                else:
                    labels = km.fit_predict(scaled_features)
                    eval_labels = labels[eval_idx] if eval_idx is not None else labels

                if len(set(eval_labels)) > 1:
                    score = silhouette_score(eval_features, eval_labels)
                    if score > best_score:
                        best_score = score
                        best_k = k

            # Fit final model with best_k
            if n_samples > 25000:
                final_km = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=2048, n_init=3)
                final_km.fit(eval_features if eval_idx is not None else scaled_features)
                cluster_labels = final_km.predict(scaled_features)
            else:
                final_km = KMeans(n_clusters=best_k, random_state=42, n_init=5)
                cluster_labels = final_km.fit_predict(scaled_features)

            # 2D PCA for visualization
            pca = PCA(n_components=2)
            # Sample for PCA visualization to prevent massive JSON transfer to Plotly
            viz_sample_size = min(n_samples, 2500)
            viz_idx = rng.choice(n_samples, size=viz_sample_size, replace=False) if n_samples > viz_sample_size else None
            
            if viz_idx is not None:
                viz_scaled = scaled_features[viz_idx]
                viz_labels = cluster_labels[viz_idx]
            else:
                viz_scaled = scaled_features
                viz_labels = cluster_labels

            pca_coords = pca.fit_transform(viz_scaled)

            # Build cluster profiles on full dataset
            clustered_df = features_df.copy()
            clustered_df["Cluster"] = [f"Cluster {l + 1}" for l in cluster_labels]
            cluster_profiles = []
            
            cluster_counts = pd.Series(cluster_labels).value_counts().to_dict()
            global_mean = features_df[self.numeric_cols].mean()

            for cluster_id in range(best_k):
                c_name = f"Cluster {cluster_id + 1}"
                count = cluster_counts.get(cluster_id, 0)
                pct = round((count / len(features_df)) * 100, 1)
                
                # Get mean characteristics of this cluster
                cluster_subset = clustered_df[clustered_df["Cluster"] == c_name][self.numeric_cols]
                c_mean = cluster_subset.mean() if len(cluster_subset) > 0 else global_mean
                
                distinguishing = []
                for col in self.numeric_cols:
                    if global_mean[col] != 0:
                        diff = (c_mean[col] - global_mean[col]) / global_mean[col]
                        if diff > 0.15:
                            distinguishing.append(f"High {col}")
                        elif diff < -0.15:
                            distinguishing.append(f"Low {col}")

                cluster_profiles.append({
                    "name": c_name,
                    "size": int(count),
                    "percentage": pct,
                    "key_traits": ", ".join(distinguishing[:3]) if distinguishing else "Moderate across metrics"
                })

            return {
                "applicable": True,
                "n_clusters": best_k,
                "silhouette_score": round(float(best_score), 3),
                "explained_variance_ratio": [round(float(v), 3) for v in pca.explained_variance_ratio_],
                "profiles": cluster_profiles,
                "pca_x": pca_coords[:, 0].tolist(),
                "pca_y": pca_coords[:, 1].tolist(),
                "labels": [f"Cluster {l + 1}" for l in viz_labels]
            }
        except Exception:
            return None

    def compute_temporal_trends(self) -> Optional[Dict[str, Any]]:
        """
        Analyzes time-series patterns if Date + Numeric columns exist.
        """
        if not self.date_cols or not self.numeric_cols:
            return None

        primary_date = self.date_cols[0]
        primary_num = self.numeric_cols[0]

        temp_df = self.df[[primary_date, primary_num]].copy()
        temp_df[primary_date] = pd.to_datetime(temp_df[primary_date], errors="coerce")
        temp_df[primary_num] = pd.to_numeric(temp_df[primary_num], errors="coerce")
        temp_df = temp_df.dropna().sort_values(by=primary_date)

        if len(temp_df) < 5:
            return None

        # Monthly aggregation
        temp_df["YearMonth"] = temp_df[primary_date].dt.to_period("M").astype(str)
        monthly = temp_df.groupby("YearMonth")[primary_num].agg(["sum", "mean", "count"]).reset_index()

        peak_month = monthly.loc[monthly["sum"].idxmax()]["YearMonth"] if len(monthly) > 0 else "N/A"
        lowest_month = monthly.loc[monthly["sum"].idxmin()]["YearMonth"] if len(monthly) > 0 else "N/A"

        # Direction calculation
        first_half = temp_df.iloc[:len(temp_df)//2][primary_num].mean()
        second_half = temp_df.iloc[len(temp_df)//2:][primary_num].mean()
        trend_pct = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0

        if trend_pct > 5:
            direction = "Upward / Growing"
        elif trend_pct < -5:
            direction = "Downward / Declining"
        else:
            direction = "Stable / Flat"

        return {
            "date_column": primary_date,
            "metric_column": primary_num,
            "overall_direction": direction,
            "trend_percentage": round(trend_pct, 1),
            "peak_period": peak_month,
            "lowest_period": lowest_month,
            "period_count": len(monthly)
        }

    def run_full_analysis(self) -> Dict[str, Any]:
        """Runs all analytical modules and bundles results."""
        return {
            "descriptive_stats": self.compute_descriptive_stats(),
            "categorical_analysis": self.compute_categorical_analysis(),
            "correlations": self.compute_correlations(),
            "outliers": self.detect_outliers(),
            "clustering": self.compute_clustering(),
            "temporal_trends": self.compute_temporal_trends()
        }
