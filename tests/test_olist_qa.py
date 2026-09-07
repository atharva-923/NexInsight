import os
import sys
import unittest

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.batch_manager import BatchManager
from core.qa_engine import DataQAEngine
from core.rag_retriever import RAGRetriever

class TestOlistQA(unittest.TestCase):

    def test_olist_products_qa(self):
        csv_path = r"C:\Users\Atharva\OneDrive\Desktop\Projects\crate\server\csv-data\olist_products_dataset.csv"
        if not os.path.exists(csv_path):
            print("Skipping olist QA test (file not found)")
            return

        ds = BatchManager.process_single_file(csv_path, "olist_products_dataset.csv")
        self.assertEqual(ds["status"], "success")

        qa = DataQAEngine(
            ds["cleaned_df"],
            ds["column_types"],
            ds["analysis_results"],
            ds["health_metrics"],
            dataset_record=ds
        )

        # 1. Total rows
        res_rows = qa.answer_query("How many rows are in this dataset?")
        self.assertIn("32,951", res_rows["answer"])
        print(f"PASS Olist: Rows question -> {res_rows['answer']}")

        # 2. Average product weight
        res_avg = qa.answer_query("What is the average product_weight_g?")
        self.assertIn("product_weight_g", res_avg["answer"])
        self.assertIsNotNone(res_avg["metric_highlight"])
        print(f"PASS Olist: Average question -> {res_avg['metric_highlight']}")

        # 3. Anomalies
        res_anom = qa.answer_query("How many anomalies were detected?")
        self.assertIn("anomalies", res_anom["answer"].lower())
        print(f"PASS Olist: Anomalies question -> {res_anom['answer'][:120]}...")

        # 4. RAG context check
        ctx = RAGRetriever.build_grounded_context(ds, "What are the correlations in products?")
        self.assertIn("olist_products_dataset.csv", ctx)
        self.assertTrue(len(ctx) < 4000)
        print("PASS Olist: RAG context successfully constructed")

if __name__ == "__main__":
    unittest.main()
