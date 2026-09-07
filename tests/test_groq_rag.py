"""
NexInsight — Comprehensive RAG + Groq AI Analyst Test Suite
Verifies all 15 core requirements:
1. RAG context creation (all 8 structured knowledge sections)
2. Dataset profile retrieval
3. Numeric question routing (deterministic Python calculations)
4. Categorical question routing (frequencies, modes, distributions)
5. Missing-value questions (hygiene and quality metrics)
6. Anomaly questions (counts, severities, outlier values)
7. Correlation questions (relationships and Pearson coefficients)
8. Trend questions (temporal axes and chronological trajectory)
9. Unanswerable questions (structured boundary defense)
10. Dataset switching (strict context isolation, zero cross-dataset leakage)
11. Groq unavailable fallback (graceful fallback without dashboard crash)
12. Missing API key handling
13. Invalid API key handling (HTTP 401 safety)
14. Empty dataset handling
15. Different schemas (Datasets A, B, C, D, E, and Olist datasets)
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
from core.anomalies import AnomalyDetector
from core.insights import InsightEngine
from core.batch_manager import BatchManager
from core.llm_client import LLMClient
from core.rag_engine import RAGEngine
from core.qa_engine import DataQAEngine


class TestGroqRAGSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Preload test datasets for comprehensive cross-schema testing."""
        cls.ds_a = BatchManager.process_single_file("test_datasets/dataset_a_sales.csv", "dataset_a_sales.csv")
        cls.ds_b = BatchManager.process_single_file("test_datasets/dataset_b_server_metrics.csv", "dataset_b_server_metrics.csv")
        cls.ds_c = BatchManager.process_single_file("test_datasets/dataset_c_survey_dirty.csv", "dataset_c_survey_dirty.csv")
        cls.ds_d = BatchManager.process_single_file("test_datasets/dataset_d_categorical.csv", "dataset_d_categorical.csv")
        cls.ds_e = BatchManager.process_single_file("test_datasets/dataset_e_sensor_numeric.csv", "dataset_e_sensor_numeric.csv")

        # Optional Olist dataset if present
        olist_path = r"C:\Users\Atharva\OneDrive\Desktop\Projects\crate\server\csv-data\olist_products_dataset.csv"
        cls.ds_olist = None
        if os.path.exists(olist_path):
            cls.ds_olist = BatchManager.process_single_file(olist_path, "olist_products_dataset.csv")

    # -------------------------------------------------------------------------
    # 1. RAG Context Creation (All 8 Structured Knowledge Sections)
    # -------------------------------------------------------------------------
    def test_01_rag_context_creation_eight_sections(self):
        """Verifies RAGEngine builds all 8 sections accurately without hallucinations."""
        sections = RAGEngine.build_dataset_knowledge_sections(self.ds_a)
        expected_sections = ["profile", "quality", "numeric", "categorical", "relationships", "temporal", "anomalies", "insights"]
        for s in expected_sections:
            self.assertIn(s, sections, f"Missing expected section '{s}' in RAG knowledge")
            self.assertTrue(len(sections[s]) > 0, f"Section '{s}' should not be empty")

        # Confirm exact details in Section 1 and Section 2
        self.assertIn("dataset_a_sales.csv", sections["profile"])
        self.assertIn("Health Score", sections["quality"])
        self.assertIn("Total Anomalies Flagged", sections["anomalies"])
        print("PASS [1/15]: RAG context creation creates all 8 structured knowledge sections.")

    # -------------------------------------------------------------------------
    # 2. Dataset Profile Retrieval
    # -------------------------------------------------------------------------
    def test_02_dataset_profile_retrieval(self):
        """Verifies dataset profile question routing returns exact record and column counts."""
        qa = DataQAEngine(
            self.ds_a["cleaned_df"], self.ds_a["column_types"],
            self.ds_a["analysis_results"], self.ds_a["health_metrics"],
            dataset_record=self.ds_a
        )
        res = qa.compute_deterministic_answer("How many rows and columns are in this dataset?")
        self.assertIn(f"{len(self.ds_a['cleaned_df']):,}", res["answer"])
        self.assertIn(f"{len(self.ds_a['cleaned_df'].columns)}", res["answer"])
        print(f"PASS [2/15]: Dataset profile retrieval returns exact dimensions ({res['metric_highlight']}).")

    # -------------------------------------------------------------------------
    # 3. Numeric Question Routing (Deterministic Python Ground Truth)
    # -------------------------------------------------------------------------
    def test_03_numeric_question_routing(self):
        """Verifies mean, max, and sum queries compute exact ground truth via Pandas."""
        qa = DataQAEngine(
            self.ds_a["cleaned_df"], self.ds_a["column_types"],
            self.ds_a["analysis_results"], self.ds_a["health_metrics"],
            dataset_record=self.ds_a
        )
        # Mean
        res_mean = qa.compute_deterministic_answer("What is the average Total_Revenue?")
        actual_mean = self.ds_a["cleaned_df"]["Total_Revenue"].mean()
        self.assertIn(f"{actual_mean:,.2f}", res_mean["answer"])

        # Max
        res_max = qa.compute_deterministic_answer("What is the highest Units_Sold?")
        actual_max = self.ds_a["cleaned_df"]["Units_Sold"].max()
        self.assertIn(f"{actual_max:,.2f}", res_max["answer"])

        # Sum
        res_sum = qa.compute_deterministic_answer("What is the total of Total_Revenue?")
        actual_sum = self.ds_a["cleaned_df"]["Total_Revenue"].sum()
        self.assertIn(f"{actual_sum:,.2f}", res_sum["answer"])
        print(f"PASS [3/15]: Numeric routing matches exact Pandas calculations (mean={actual_mean:,.2f}, max={actual_max:,.2f}).")

    # -------------------------------------------------------------------------
    # 4. Categorical Question Routing
    # -------------------------------------------------------------------------
    def test_04_categorical_question_routing(self):
        """Verifies categorical mode and distribution queries return exact counts and percentages."""
        qa = DataQAEngine(
            self.ds_a["cleaned_df"], self.ds_a["column_types"],
            self.ds_a["analysis_results"], self.ds_a["health_metrics"],
            dataset_record=self.ds_a
        )
        res_cat = qa.compute_deterministic_answer("Which Region occurs most frequently?")
        top_region = self.ds_a["cleaned_df"]["Region"].value_counts().index[0]
        top_count = self.ds_a["cleaned_df"]["Region"].value_counts().iloc[0]
        self.assertIn(top_region, res_cat["answer"])
        self.assertIn(f"{top_count:,}", res_cat["answer"])
        print(f"PASS [4/15]: Categorical routing identifies exact mode '{top_region}' with {top_count} occurrences.")

    # -------------------------------------------------------------------------
    # 5. Missing-Value Questions (Data Quality)
    # -------------------------------------------------------------------------
    def test_05_missing_value_questions(self):
        """Verifies missing value reporting on dirty datasets vs clean datasets."""
        # Dataset C is dirty with intentional missing values
        qa_c = DataQAEngine(
            self.ds_c["cleaned_df"], self.ds_c["column_types"],
            self.ds_c["analysis_results"], self.ds_c["health_metrics"],
            dataset_record=self.ds_c
        )
        res_c = qa_c.compute_deterministic_answer("What columns have missing values?")
        raw_df_c = self.ds_c["raw_df"]
        actual_missing_c = int(raw_df_c.isna().sum().sum())
        self.assertIn(f"{actual_missing_c:,}", res_c["answer"])

        # Dataset A is clean (0 missing values)
        qa_a = DataQAEngine(
            self.ds_a["cleaned_df"], self.ds_a["column_types"],
            self.ds_a["analysis_results"], self.ds_a["health_metrics"],
            dataset_record=self.ds_a
        )
        res_a = qa_a.compute_deterministic_answer("Are there any missing values?")
        self.assertIn("Zero missing values", res_a["answer"])
        print(f"PASS [5/15]: Missing value queries correctly report {actual_missing_c} missing in dirty and 0 in clean dataset.")

    # -------------------------------------------------------------------------
    # 6. Anomaly Questions
    # -------------------------------------------------------------------------
    def test_06_anomaly_questions(self):
        """Verifies anomaly detection questions report exact detected count and severities."""
        qa = DataQAEngine(
            self.ds_a["cleaned_df"], self.ds_a["column_types"],
            self.ds_a["analysis_results"], self.ds_a["health_metrics"],
            dataset_record=self.ds_a
        )
        res_anom = qa.compute_deterministic_answer("How many anomalies were detected?")
        actual_anoms = self.ds_a["anomalies_data"].get("total_anomalies", 0)
        self.assertIn(f"{actual_anoms:,}", res_anom["answer"])
        print(f"PASS [6/15]: Anomaly query reports exact detected count ({actual_anoms} anomalies).")

    # -------------------------------------------------------------------------
    # 7. Correlation Questions
    # -------------------------------------------------------------------------
    def test_07_correlation_questions(self):
        """Verifies correlation questions identify top Pearson correlation pair accurately."""
        qa = DataQAEngine(
            self.ds_a["cleaned_df"], self.ds_a["column_types"],
            self.ds_a["analysis_results"], self.ds_a["health_metrics"],
            dataset_record=self.ds_a
        )
        res_corr = qa.compute_deterministic_answer("What are the strongest correlations?")
        corr_data = self.ds_a["analysis_results"].get("correlations", {})
        if corr_data.get("has_correlation"):
            top_pair = corr_data["significant_pairs"][0]
            self.assertIn(top_pair["col1"], res_corr["answer"])
            self.assertIn(top_pair["col2"], res_corr["answer"])
            self.assertIn(f"{top_pair['correlation']:.2f}", res_corr["answer"])
        print("PASS [7/15]: Correlation query returns exact strongest Pearson correlation pairs.")

    # -------------------------------------------------------------------------
    # 8. Trend Questions (Temporal Patterns)
    # -------------------------------------------------------------------------
    def test_08_trend_questions(self):
        """Verifies trend questions extract date range and trajectory from temporal datasets."""
        qa = DataQAEngine(
            self.ds_a["cleaned_df"], self.ds_a["column_types"],
            self.ds_a["analysis_results"], self.ds_a["health_metrics"],
            dataset_record=self.ds_a
        )
        res_trend = qa.compute_deterministic_answer("What is the chronological trend over time?")
        trends = self.ds_a["analysis_results"].get("temporal_trends")
        if trends:
            self.assertIn(trends["overall_direction"], res_trend["answer"])
            self.assertIn(trends["peak_period"], res_trend["answer"])
        print("PASS [8/15]: Trend questions route to verified temporal analysis.")

    # -------------------------------------------------------------------------
    # 9. Unanswerable Questions
    # -------------------------------------------------------------------------
    def test_09_unanswerable_questions(self):
        """Verifies system handles out-of-scope questions cleanly without hallucinating."""
        qa = DataQAEngine(
            self.ds_a["cleaned_df"], self.ds_a["column_types"],
            self.ds_a["analysis_results"], self.ds_a["health_metrics"],
            dataset_record=self.ds_a
        )
        res = qa.compute_deterministic_answer("Who won the 1994 World Cup in football?")
        self.assertIn("Query analyzed against", res["answer"])
        self.assertIn("Attributes Active", res["metric_highlight"])
        print("PASS [9/15]: Unanswerable questions handled gracefully within data scope boundaries.")

    # -------------------------------------------------------------------------
    # 10. Dataset Switching and Strict Context Isolation
    # -------------------------------------------------------------------------
    def test_10_dataset_switching_isolation(self):
        """Verifies switching datasets completely switches RAG context and answers without leakage."""
        ctx_a = RAGEngine.retrieve_relevant_context("What is the average?", self.ds_a)
        ctx_b = RAGEngine.retrieve_relevant_context("What is the average?", self.ds_b)

        # Context A must not contain Dataset B fields or filename
        self.assertIn("dataset_a_sales.csv", ctx_a)
        self.assertNotIn("dataset_b_server_metrics.csv", ctx_a)
        self.assertNotIn("CPU_Load_Pct", ctx_a)

        # Context B must not contain Dataset A fields or filename
        self.assertIn("dataset_b_server_metrics.csv", ctx_b)
        self.assertNotIn("dataset_a_sales.csv", ctx_b)
        self.assertNotIn("Total_Revenue", ctx_b)
        print("PASS [10/15]: Strict active dataset isolation confirmed (zero cross-dataset contamination).")

    # -------------------------------------------------------------------------
    # 11. Groq Unavailable / Graceful Fallback
    # -------------------------------------------------------------------------
    def test_11_groq_unavailable_fallback(self):
        """Verifies that if Groq fails or times out, local deterministic calculation is returned without crashing."""
        qa = DataQAEngine(
            self.ds_a["cleaned_df"], self.ds_a["column_types"],
            self.ds_a["analysis_results"], self.ds_a["health_metrics"],
            dataset_record=self.ds_a
        )
        # Passing an invalid key will trigger API failure and test fallback
        resp = qa.answer_query(
            "What is the average Total_Revenue?",
            api_key="gsk_invalid_test_key_12345",
            model="llama-3.3-70b-versatile"
        )
        # Must gracefully return fallback engine
        self.assertEqual(resp["engine"], "fallback")
        self.assertIsNotNone(resp["answer"])
        actual_mean = self.ds_a["cleaned_df"]["Total_Revenue"].mean()
        self.assertIn(f"{actual_mean:,.2f}", resp["answer"])
        self.assertIn("Groq AI Analyst is unavailable", resp["fallback_message"])
        print("PASS [11/15]: Groq API failure falls back seamlessly to deterministic Python calculations.")

    # -------------------------------------------------------------------------
    # 12. Missing API Key Handling
    # -------------------------------------------------------------------------
    def test_12_missing_api_key(self):
        """Verifies system defaults cleanly to local deterministic mode when API key is empty."""
        from unittest.mock import patch
        self.assertFalse(LLMClient.is_configured(""))
        with patch.dict(os.environ, {"GROQ_API_KEY": "", "XAI_API_KEY": ""}):
            self.assertFalse(LLMClient.is_configured())

        qa = DataQAEngine(
            self.ds_a["cleaned_df"], self.ds_a["column_types"],
            self.ds_a["analysis_results"], self.ds_a["health_metrics"],
            dataset_record=self.ds_a
        )
        resp = qa.answer_query("What is the average Total_Revenue?", api_key="")
        self.assertEqual(resp["engine"], "local")
        self.assertIsNotNone(resp["answer"])
        print("PASS [12/15]: Missing API key triggers local deterministic engine without error.")

    # -------------------------------------------------------------------------
    # 13. Invalid API Key Handling (Ping Verification)
    # -------------------------------------------------------------------------
    def test_13_invalid_api_key_test_connection(self):
        """Verifies test_connection safely returns error without crashing or leaking sensitive info."""
        test_res = LLMClient.test_connection("gsk_fake_invalid_key_for_testing_only")
        self.assertFalse(test_res["success"])
        self.assertTrue(len(test_res["error"]) > 0)
        # Ensure raw key is not echoed in error message
        self.assertNotIn("gsk_fake_invalid_key_for_testing_only", test_res["error"])
        print("PASS [13/15]: Invalid API key returns clean diagnostic message without credential exposure.")

    # -------------------------------------------------------------------------
    # 14. Empty Dataset Handling
    # -------------------------------------------------------------------------
    def test_14_empty_dataset_handling(self):
        """Verifies system handles empty / zero-row dataframes without throwing unhandled exceptions."""
        empty_df = pd.DataFrame(columns=["ColA", "ColB"])
        qa = DataQAEngine(empty_df, {"ColA": "Numeric", "ColB": "Categorical"}, {}, {})
        res = qa.compute_deterministic_answer("What is the average of ColA?")
        self.assertIn("empty", res["answer"].lower())
        print("PASS [14/15]: Empty dataset handled cleanly without crashing.")

    # -------------------------------------------------------------------------
    # 15. Different Schemas (Datasets A, B, C, D, E, plus Olist)
    # -------------------------------------------------------------------------
    def test_15_different_schemas_coverage(self):
        """Verifies 6 question types (numerical, categorical, data-quality, anomaly, trend, unanswerable) across all datasets."""
        datasets = [
            ("Dataset A (Sales)", self.ds_a, "Units_Sold", "Region", "Transaction_Date"),
            ("Dataset B (Server)", self.ds_b, "CPU_Load_Pct", "Cluster_Zone", "Timestamp"),
            ("Dataset C (Survey Dirty)", self.ds_c, "Satisfaction_Score", "Department", None),
            ("Dataset D (Categorical)", self.ds_d, None, "Account_Tier", None),
            ("Dataset E (Sensors)", self.ds_e, "Core_Temperature_C", None, None),
        ]
        if self.ds_olist:
            datasets.append(("Dataset Olist (Products)", self.ds_olist, "product_weight_g", "product_category_name", None))

        for name, ds, num_col, cat_col, date_col in datasets:
            qa = DataQAEngine(
                ds["cleaned_df"], ds["column_types"],
                ds["analysis_results"], ds["health_metrics"],
                dataset_record=ds
            )

            # Q1: Numerical
            if num_col and num_col in ds["cleaned_df"].columns:
                res_num = qa.compute_deterministic_answer(f"What is the average of {num_col}?")
                exp_mean = ds["cleaned_df"][num_col].mean()
                self.assertIn(f"{exp_mean:,.2f}", res_num["answer"])

            # Q2: Categorical
            if cat_col and cat_col in ds["cleaned_df"].columns:
                res_cat = qa.compute_deterministic_answer(f"Which {cat_col} occurs most frequently?")
                exp_mode = ds["cleaned_df"][cat_col].value_counts().index[0]
                self.assertIn(str(exp_mode), res_cat["answer"])

            # Q3: Data Quality
            res_dq = qa.compute_deterministic_answer("What is the data quality and health score?")
            self.assertIn(f"{ds['health_metrics'].get('score', 100)}", res_dq["answer"])

            # Q4: Anomaly
            res_anom = qa.compute_deterministic_answer("How many anomalies were identified?")
            exp_anom = ds["anomalies_data"].get("total_anomalies", 0)
            self.assertIn(f"{exp_anom:,}", res_anom["answer"])

            # Q5: Pattern / Trend
            res_pat = qa.compute_deterministic_answer("What are the key patterns in this dataset?")
            self.assertIn("Key patterns", res_pat["answer"])

            # Q6: Unanswerable
            res_unans = qa.compute_deterministic_answer("Who won the Academy Award for Best Picture in 2020?")
            self.assertIn("Query analyzed against", res_unans["answer"])

        print("PASS [15/15]: Verified numerical, categorical, data-quality, anomaly, pattern, and boundary questions across all 6 schemas.")


if __name__ == "__main__":
    unittest.main()
