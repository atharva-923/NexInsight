"""
End-to-end integration test verifying the Editorial Enterprise UI logic,
data extraction, and calculations across all datasets.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.data_processor import DataProcessor
from core.analyzer import DataAnalyzer
from core.visualizer import Visualizer
from core.insights import InsightEngine
from core.anomalies import AnomalyDetector
from core.qa_engine import DataQAEngine
from core.batch_manager import BatchManager


class TestEditorialUI(unittest.TestCase):

    def setUp(self):
        self.datasets = {}
        for fname in ["dataset_a_sales.csv", "dataset_b_server_metrics.csv", "dataset_c_survey_dirty.csv", "dataset_d_categorical.csv", "dataset_e_sensor_numeric.csv"]:
            path = os.path.join("test_datasets", fname)
            if os.path.exists(path):
                res = BatchManager.process_single_file(path, fname)
                self.datasets[res["id"]] = res

    def test_overview_kpis_and_charts(self):
        """Verify Overview KPI metrics and charts on all datasets."""
        for ds_id, ds in self.datasets.items():
            df = ds["cleaned_df"]
            raw_df = ds["raw_df"]
            h = ds["health_metrics"]
            clean_sum = ds["cleaning_summary"]
            anoms = ds["anomalies_data"]
            col_types = ds["column_types"]

            # KPIs
            self.assertTrue(0 <= h["score"] <= 100)
            self.assertEqual(len(df), ds["cleaned_row_count"])
            self.assertEqual(len(df.columns), len(col_types))
            self.assertIn("total_missing", h)
            self.assertIn("total_anomalies", anoms)

            # Auto charts
            charts = Visualizer.auto_generate_dashboard_charts(df, col_types)
            self.assertIsInstance(charts, list)
            if len(charts) > 0:
                self.assertIsNotNone(charts[0])

    def test_explore_schema_and_stats(self):
        """Verify Explore schema generation and data types."""
        for ds_id, ds in self.datasets.items():
            df = ds["cleaned_df"]
            col_types = ds["column_types"]
            for col, ctype in col_types.items():
                self.assertIn(ctype, ["Numeric", "Categorical", "Date", "Boolean", "Text", "ID"])
                self.assertIn(col, df.columns)

    def test_patterns_and_clusters(self):
        """Verify Patterns page insights, correlations, and clustering."""
        for ds_id, ds in self.datasets.items():
            insights = ds["ai_insights"]
            analysis = ds["analysis_results"]

            self.assertIn("executive_summary", insights)
            self.assertIn("key_findings", insights)
            self.assertIn("correlations", analysis)

            clustering = analysis.get("clustering")
            if clustering and clustering.get("applicable"):
                self.assertIn("pca_x", clustering)
                self.assertIn("pca_y", clustering)
                self.assertIn("profiles", clustering)

    def test_outliers_reporting(self):
        """Verify Outliers page anomaly data integrity."""
        for ds_id, ds in self.datasets.items():
            anoms = ds["anomalies_data"]
            self.assertIn("severity_counts", anoms)
            crit = anoms["severity_counts"].get("Critical", 0)
            mod = anoms["severity_counts"].get("Moderate", 0)
            low = anoms["severity_counts"].get("Low", 0)
            self.assertEqual(anoms["total_anomalies"], crit + mod + low)

    def test_ask_command_interface(self):
        """Verify Ask command interface produces verified responses with metric highlights."""
        ds_a = next(ds for ds in self.datasets.values() if "sales" in ds["filename"].lower())
        qa = DataQAEngine(
            ds_a["cleaned_df"],
            ds_a["column_types"],
            ds_a["analysis_results"],
            ds_a["health_metrics"],
            dataset_record=ds_a
        )
        res = qa.answer_query("What is the average revenue?")
        self.assertIn("answer", res)
        self.assertIsNotNone(res.get("metric_highlight"))


if __name__ == "__main__":
    unittest.main()
