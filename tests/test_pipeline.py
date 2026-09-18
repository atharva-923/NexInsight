"""
Comprehensive End-to-End Pipeline Tests for NexInsight
Validates all 5 diverse test datasets and edge cases without assumptions.

Test data is generated synthetically in setUpClass; no committed CSV/XLSX files required.
"""

import os
import sys
import tempfile
import shutil
import unittest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.fixtures import (
    make_dataset_a, make_dataset_b, make_dataset_c,
    make_dataset_d, make_dataset_e, write_csv,
)
from core.data_processor import DataProcessor
from core.analyzer import DataAnalyzer
from core.visualizer import Visualizer
from core.insights import InsightEngine
from core.anomalies import AnomalyDetector
from core.qa_engine import DataQAEngine


class TestNexInsightPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Write synthetic datasets A-E to a temp directory."""
        cls._tmpdir = tempfile.mkdtemp(prefix="nexinsight_test_")
        
        cls.csv_a = os.path.join(cls._tmpdir, "dataset_a_sales.csv")
        cls.csv_b = os.path.join(cls._tmpdir, "dataset_b_server_metrics.csv")
        cls.csv_c = os.path.join(cls._tmpdir, "dataset_c_survey_dirty.csv")
        cls.csv_d = os.path.join(cls._tmpdir, "dataset_d_categorical.csv")
        cls.csv_e = os.path.join(cls._tmpdir, "dataset_e_sensor_numeric.csv")
        
        write_csv(make_dataset_a(), cls.csv_a)
        write_csv(make_dataset_b(), cls.csv_b)
        write_csv(make_dataset_c(), cls.csv_c)
        write_csv(make_dataset_d(), cls.csv_d)
        write_csv(make_dataset_e(), cls.csv_e)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def setUp(self):
        self.processor = DataProcessor()

    def _run_full_pipeline_check(self, filepath: str, label: str):
        """Runs the entire pipeline end-to-end on a given dataset file."""
        self.assertTrue(os.path.exists(filepath), f"File {filepath} must exist")
        filename = os.path.basename(filepath)

        # 1. Processing & Cleaning
        res = self.processor.process(filepath, filename)
        self.assertIsNotNone(res["cleaned_df"])
        self.assertIsNotNone(res["column_types"])
        self.assertIn("score", res["health_metrics"])
        self.assertTrue(0 <= res["health_metrics"]["score"] <= 100)

        cleaned_df = res["cleaned_df"]
        col_types = res["column_types"]
        health = res["health_metrics"]
        clean_sum = res["cleaning_summary"]

        # 2. Statistical Analysis
        analyzer = DataAnalyzer(cleaned_df, col_types)
        analysis = analyzer.run_full_analysis()
        self.assertIn("descriptive_stats", analysis)
        self.assertIn("categorical_analysis", analysis)
        self.assertIn("correlations", analysis)
        self.assertIn("outliers", analysis)

        # 3. Anomaly Detection
        detector = AnomalyDetector(cleaned_df, col_types)
        anomalies = detector.get_comprehensive_anomalies()
        self.assertIn("total_anomalies", anomalies)
        self.assertIn("anomalies_list", anomalies)

        # 4. Insights Generation
        insights = InsightEngine.generate_factual_insights(analysis, health, clean_sum)
        self.assertIn("executive_summary", insights)
        self.assertTrue(len(insights["executive_summary"]) > 10)
        self.assertIn("key_findings", insights)

        # 5. Automated Charts
        charts = Visualizer.auto_generate_dashboard_charts(cleaned_df, col_types)
        self.assertIsInstance(charts, list)

        # 6. Natural Language QA Engine
        qa = DataQAEngine(cleaned_df, col_types, analysis, health)
        test_queries = [
            "What is the highest-performing category?",
            "Are there unusual records?",
            "What are the strongest correlations?",
            "What are the most important patterns in this dataset?"
        ]
        for q in test_queries:
            ans = qa.answer_query(q)
            self.assertIn("answer", ans)
            self.assertTrue(len(ans["answer"]) > 5)

        print(f"PASS: {label} ({filename}) - {len(cleaned_df)} rows, Health: {health['score']}/100, Charts: {len(charts)}")

    def test_dataset_a_sales(self):
        self._run_full_pipeline_check(self.csv_a, "Dataset A (Sales)")

    def test_dataset_b_servers(self):
        self._run_full_pipeline_check(self.csv_b, "Dataset B (Server Metrics)")

    def test_dataset_c_survey_dirty(self):
        # Must remove duplicates and impute missing
        self._run_full_pipeline_check(self.csv_c, "Dataset C (Dirty Survey)")
        res = self.processor.process(self.csv_c, "dataset_c_survey_dirty.csv")
        self.assertTrue(res["cleaning_summary"]["duplicates_removed"] > 0)
        self.assertTrue(res["cleaning_summary"]["missing_values_handled"] > 0)

    def test_dataset_d_categorical(self):
        # Must not crash when there are NO numerical columns or dates
        self._run_full_pipeline_check(self.csv_d, "Dataset D (Categorical)")

    def test_dataset_e_sensors(self):
        # Must not crash when there are NO categorical columns or dates
        self._run_full_pipeline_check(self.csv_e, "Dataset E (Numeric Sensors)")

    def test_edge_case_minimal(self):
        # Test minimal 2-row dataframe
        df = pd.DataFrame({"Value": [10.5, 20.2], "Tag": ["A", "B"]})
        col_types = self.processor.detect_column_types(df)
        cleaned, summary = self.processor.clean_dataset(df)
        analyzer = DataAnalyzer(cleaned, col_types)
        analysis = analyzer.run_full_analysis()
        self.assertIsNotNone(analysis)
        print("PASS: Minimal Edge Case")


if __name__ == "__main__":
    unittest.main()
