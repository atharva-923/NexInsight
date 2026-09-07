import os
import sys
import time
import glob
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.data_processor import DataProcessor
from core.analyzer import DataAnalyzer
from core.anomalies import AnomalyDetector
from core.insights import InsightEngine

data_dir = r"C:\Users\Atharva\OneDrive\Desktop\Projects\crate\server\csv-data"
csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

def log_print(msg):
    print(msg, flush=True)

log_print(f"Found {len(csv_files)} files in {data_dir}:")
for f in csv_files:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    log_print(f"  {os.path.basename(f)} ({size_mb:.2f} MB)")

processor = DataProcessor()

for filepath in csv_files:
    fname = os.path.basename(filepath)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    log_print(f"\n--- Testing: {fname} ({size_mb:.2f} MB) ---")
    
    # 1. Loading
    t0 = time.time()
    log_print("Starting loading...")
    raw_df = processor.load_file(filepath, fname)
    t_load = time.time() - t0
    log_print(f"Loading: {t_load:.2f}s (rows: {len(raw_df):,}, cols: {len(raw_df.columns)})")
    
    # 2. Type detection
    t0 = time.time()
    log_print("Starting type detection...")
    col_types = processor.detect_column_types(raw_df)
    t_typedetect = time.time() - t0
    log_print(f"Type detection: {t_typedetect:.2f}s")
    
    # 3. Cleaning
    t0 = time.time()
    log_print("Starting cleaning...")
    cleaned_df, clean_summary = processor.clean_dataset(raw_df, remove_duplicates=True)
    health_metrics = processor.calculate_health_score(raw_df, cleaned_df)
    t_clean = time.time() - t0
    log_print(f"Cleaning: {t_clean:.2f}s (cleaned rows: {len(cleaned_df):,})")
    
    analyzer = DataAnalyzer(cleaned_df, col_types)
    
    # 4. Statistics
    t0 = time.time()
    log_print("Starting statistics...")
    desc_stats = analyzer.compute_descriptive_stats()
    cat_stats = analyzer.compute_categorical_analysis()
    t_stats = time.time() - t0
    log_print(f"Statistics: {t_stats:.2f}s")
    
    # 5. Correlations
    t0 = time.time()
    log_print("Starting correlation...")
    corr_results = analyzer.compute_correlations()
    t_corr = time.time() - t0
    log_print(f"Correlation: {t_corr:.2f}s")
    
    # 6. Anomalies
    t0 = time.time()
    log_print("Starting anomalies...")
    detector = AnomalyDetector(cleaned_df, col_types)
    anomalies_data = detector.get_comprehensive_anomalies()
    t_anom = time.time() - t0
    log_print(f"Anomalies: {t_anom:.2f}s (detected: {anomalies_data['total_anomalies']:,})")
    
    # 7. Clustering
    t0 = time.time()
    log_print("Starting clustering...")
    clustering = analyzer.compute_clustering()
    t_cluster = time.time() - t0
    log_print(f"Clustering: {t_cluster:.2f}s (k={clustering.get('n_clusters') if clustering else 'None'})")
    
    # 8. Insights
    t0 = time.time()
    log_print("Starting insights...")
    analysis_results = {
        "descriptive_stats": desc_stats,
        "categorical_analysis": cat_stats,
        "correlations": corr_results,
        "outliers": analyzer.detect_outliers(),
        "clustering": clustering,
        "temporal_trends": analyzer.compute_temporal_trends()
    }
    ai_insights = InsightEngine.generate_factual_insights(analysis_results, health_metrics, clean_summary)
    t_insights = time.time() - t0
    log_print(f"Insights: {t_insights:.2f}s")
    
    total = t_load + t_typedetect + t_clean + t_stats + t_corr + t_anom + t_cluster + t_insights
    log_print(f"Total time for {fname}: {total:.2f}s")
