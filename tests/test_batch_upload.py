"""
Automated Test Suite for NexInsight Multi-File / Batch Upload
Covers Scenarios A through I:
A. One CSV
B. One XLSX
C. Multiple CSV files
D. Multiple XLSX files
E. CSV + XLSX together
F. One invalid file + valid files
G. Multiple files with same/similar filenames (name collision safety)
H. Existing five test datasets in batch
I. Dataset isolation and switching (zero leakage between datasets)
"""

import os
import sys
import unittest
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.batch_manager import BatchManager
from core.qa_engine import DataQAEngine


class TestBatchUpload(unittest.TestCase):

    def setUp(self):
        self.csv_a = "test_datasets/dataset_a_sales.csv"
        self.csv_b = "test_datasets/dataset_b_server_metrics.csv"
        self.csv_c = "test_datasets/dataset_c_survey_dirty.csv"
        self.csv_d = "test_datasets/dataset_d_categorical.csv"
        self.csv_e = "test_datasets/dataset_e_sensor_numeric.csv"
        self.xlsx_1 = "test_datasets/test_batch_1.xlsx"
        self.xlsx_2 = "test_datasets/test_batch_2.xlsx"
        self.invalid_empty = "test_datasets/test_invalid_empty.csv"
        self.corrupt_xlsx = "test_datasets/test_corrupt.xlsx"

    def test_scenario_a_single_csv(self):
        """A. One CSV upload"""
        datasets, logs = BatchManager.process_batch([self.csv_a])
        self.assertEqual(len(datasets), 1)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], "success")
        record = list(datasets.values())[0]
        self.assertEqual(record["filename"], "dataset_a_sales.csv")
        self.assertTrue(record["cleaned_row_count"] > 0)
        self.assertIn("score", record["health_metrics"])
        print("PASS: Scenario A - One CSV")

    def test_scenario_b_single_xlsx(self):
        """B. One XLSX upload"""
        datasets, logs = BatchManager.process_batch([self.xlsx_1])
        self.assertEqual(len(datasets), 1)
        self.assertEqual(logs[0]["status"], "success")
        record = list(datasets.values())[0]
        self.assertEqual(record["filename"], "test_batch_1.xlsx")
        self.assertEqual(record["cleaned_row_count"], 100)
        print("PASS: Scenario B - One XLSX")

    def test_scenario_c_multiple_csv(self):
        """C. Multiple CSV files"""
        files = [self.csv_a, self.csv_b, self.csv_c]
        datasets, logs = BatchManager.process_batch(files)
        self.assertEqual(len(datasets), 3)
        self.assertTrue(all(log["status"] == "success" for log in logs))
        summary_table = BatchManager.get_batch_summary_table(datasets)
        self.assertEqual(len(summary_table), 3)
        print("PASS: Scenario C - Multiple CSV files")

    def test_scenario_d_multiple_xlsx(self):
        """D. Multiple XLSX files"""
        files = [self.xlsx_1, self.xlsx_2]
        datasets, logs = BatchManager.process_batch(files)
        self.assertEqual(len(datasets), 2)
        self.assertTrue(all(log["status"] == "success" for log in logs))
        print("PASS: Scenario D - Multiple XLSX files")

    def test_scenario_e_csv_and_xlsx_together(self):
        """E. CSV + XLSX together in same batch"""
        files = [self.csv_a, self.xlsx_1]
        datasets, logs = BatchManager.process_batch(files)
        self.assertEqual(len(datasets), 2)
        self.assertTrue(all(log["status"] == "success" for log in logs))
        print("PASS: Scenario E - CSV and XLSX combined")

    def test_scenario_f_invalid_file_handling(self):
        """F. One invalid file + valid files (invalid fails gracefully without crashing batch)"""
        files = [self.csv_a, self.invalid_empty, self.corrupt_xlsx, self.csv_b]
        datasets, logs = BatchManager.process_batch(files)
        # 2 valid should succeed, 2 invalid should fail gracefully
        self.assertEqual(len(datasets), 2)
        self.assertEqual(len(logs), 4)
        successes = [l for l in logs if l["status"] == "success"]
        failures = [l for l in logs if l["status"] == "failed"]
        self.assertEqual(len(successes), 2)
        self.assertEqual(len(failures), 2)
        print("PASS: Scenario F - Fault tolerance on corrupted & empty files")

    def test_scenario_g_duplicate_filenames(self):
        """G. Multiple files with same filename (ensures unique naming and separate records)"""
        # Uploading dataset_a_sales twice
        files = [self.csv_a, self.csv_a]
        datasets, logs = BatchManager.process_batch(files)
        self.assertEqual(len(datasets), 2)
        names = [d["filename"] for d in datasets.values()]
        self.assertIn("dataset_a_sales.csv", names)
        self.assertIn("dataset_a_sales (2).csv", names)
        self.assertEqual(len(set(datasets.keys())), 2)  # Distinct internal IDs
        print("PASS: Scenario G - Duplicate filename collision resolution")

    def test_scenario_h_all_five_test_datasets(self):
        """H. Existing five test datasets processed in batch"""
        files = [self.csv_a, self.csv_b, self.csv_c, self.csv_d, self.csv_e]
        datasets, logs = BatchManager.process_batch(files)
        self.assertEqual(len(datasets), 5)
        self.assertTrue(all(log["status"] == "success" for log in logs))
        summary = BatchManager.get_batch_summary_table(datasets)
        self.assertEqual(len(summary), 5)
        print("PASS: Scenario H - All 5 core test datasets processed together")

    def test_scenario_i_dataset_isolation(self):
        """I. Strict dataset isolation - zero metric leakage between datasets"""
        files = [self.csv_a, self.csv_b]
        datasets, _ = BatchManager.process_batch(files)
        ds_a = [d for d in datasets.values() if d["filename"] == "dataset_a_sales.csv"][0]
        ds_b = [d for d in datasets.values() if d["filename"] == "dataset_b_server_metrics.csv"][0]

        # 1. Row counts are distinct
        self.assertNotEqual(ds_a["cleaned_row_count"], ds_b["cleaned_row_count"])
        # 2. Schema and columns are completely isolated
        self.assertIn("Total_Revenue", ds_a["cleaned_df"].columns)
        self.assertNotIn("Total_Revenue", ds_b["cleaned_df"].columns)
        self.assertIn("CPU_Load_Pct", ds_b["cleaned_df"].columns)
        self.assertNotIn("CPU_Load_Pct", ds_a["cleaned_df"].columns)

        # 3. Ask Your Data context isolation
        qa_a = DataQAEngine(ds_a["cleaned_df"], ds_a["column_types"], ds_a["analysis_results"], ds_a["health_metrics"])
        qa_b = DataQAEngine(ds_b["cleaned_df"], ds_b["column_types"], ds_b["analysis_results"], ds_b["health_metrics"])

        ans_a = qa_a.answer_query("What is the average revenue?")
        ans_b = qa_b.answer_query("What is the average CPU load?")

        self.assertIn("revenue", ans_a["answer"].lower())
        self.assertIn("cpu_load_pct", ans_b["answer"].lower())

        print("PASS: Scenario I - Dataset isolation and non-leakage verified")


if __name__ == "__main__":
    unittest.main()
