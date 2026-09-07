"""
NexInsight - Batch & Multi-File Dataset Manager
Handles ingestion of multiple CSV and Excel files, unique dataset isolation,
robust error recovery for corrupted/empty files, duplicate filename resolution,
per-file & per-stage progress tracking with granular stage timings,
and instantaneous switching without recomputation.
"""

import os
import time
import uuid
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
from core.data_processor import DataProcessor
from core.analyzer import DataAnalyzer
from core.anomalies import AnomalyDetector
from core.insights import InsightEngine


class BatchManager:
    """Manages multi-dataset sessions with complete state isolation and stage-level telemetry."""

    @staticmethod
    def generate_display_name(filename: str, existing_names: List[str]) -> str:
        """
        Ensures human-readable uniqueness for duplicate filenames.
        e.g., 'sales.csv' -> 'sales (2).csv' if 'sales.csv' already exists.
        """
        if filename not in existing_names:
            return filename

        name_part, ext = os.path.splitext(filename)
        counter = 2
        while True:
            new_name = f"{name_part} ({counter}){ext}"
            if new_name not in existing_names:
                return new_name
            counter += 1

    @classmethod
    def process_single_file(
        cls,
        file_obj,
        raw_filename: str,
        existing_display_names: Optional[List[str]] = None,
        remove_duplicates: bool = True,
        progress_callback: Optional[Callable[[int, int, str, str, str, float], None]] = None,
        file_index: int = 1,
        total_files: int = 1
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end pipeline on one file with strict validation and per-stage timing.
        Returns a dict containing either a valid dataset record or error details.
        """
        if existing_display_names is None:
            existing_display_names = []

        display_name = cls.generate_display_name(raw_filename, existing_display_names)
        dataset_id = f"{uuid.uuid4().hex[:8]}_{display_name}"

        stage_timings: Dict[str, float] = {
            "loading": 0.0,
            "type_detection": 0.0,
            "cleaning": 0.0,
            "statistics": 0.0,
            "correlations": 0.0,
            "anomalies": 0.0,
            "clustering": 0.0,
            "insights": 0.0,
            "total": 0.0
        }

        # 1. Extension validation
        ext = raw_filename.split(".")[-1].lower() if "." in raw_filename else ""
        if ext not in ["csv", "xlsx", "xls", "txt"]:
            return {
                "id": dataset_id,
                "filename": display_name,
                "status": "failed",
                "stages": {"loaded": False, "cleaned": False, "analyzed": False},
                "stage_timings": stage_timings,
                "error": f"Unsupported format '.{ext}'. Supported formats: CSV, XLSX, XLS."
            }

        processor = DataProcessor()
        stages = {"loaded": False, "cleaned": False, "analyzed": False}
        t_start_total = time.time()

        # Helper to notify progress callback
        def notify(stage_name: str, stage_status: str, elapsed: float = 0.0):
            if progress_callback:
                try:
                    progress_callback(file_index, total_files, display_name, stage_name, stage_status, elapsed)
                except Exception:
                    pass

        # 2. Stage: Loading
        notify("Loading", "running")
        t0 = time.time()
        try:
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            raw_df = processor.load_file(file_obj, raw_filename)
            
            if raw_df is None or raw_df.empty or len(raw_df.columns) == 0:
                stage_timings["loading"] = round(time.time() - t0, 2)
                stage_timings["total"] = round(time.time() - t_start_total, 2)
                return {
                    "id": dataset_id,
                    "filename": display_name,
                    "status": "failed",
                    "stages": stages,
                    "stage_timings": stage_timings,
                    "error": "Dataset contains zero rows or zero valid columns."
                }
            stage_timings["loading"] = round(time.time() - t0, 2)
            stages["loaded"] = True
            notify("Loading", "completed", stage_timings["loading"])
        except Exception as e:
            stage_timings["loading"] = round(time.time() - t0, 2)
            stage_timings["total"] = round(time.time() - t_start_total, 2)
            return {
                "id": dataset_id,
                "filename": display_name,
                "status": "failed",
                "stages": stages,
                "stage_timings": stage_timings,
                "error": f"Failed to parse file: {str(e)}"
            }

        # 3. Stage: Type Detection
        notify("Type Detection", "running")
        t0 = time.time()
        try:
            col_types = processor.detect_column_types(raw_df)
            stage_timings["type_detection"] = round(time.time() - t0, 2)
            notify("Type Detection", "completed", stage_timings["type_detection"])
        except Exception as e:
            col_types = {c: "Text" for c in raw_df.columns}
            stage_timings["type_detection"] = round(time.time() - t0, 2)

        # 4. Stage: Cleaning
        notify("Cleaning", "running")
        t0 = time.time()
        try:
            cleaned_df, clean_summary = processor.clean_dataset(raw_df, remove_duplicates=remove_duplicates)
            health_metrics = processor.calculate_health_score(
                raw_df, cleaned_df, duplicate_rows_count=clean_summary.get("duplicates_detected")
            )
            col_types = processor.detect_column_types(cleaned_df)
            stage_timings["cleaning"] = round(time.time() - t0, 2)
            stages["cleaned"] = True
            notify("Cleaning", "completed", stage_timings["cleaning"])
        except Exception as e:
            stage_timings["cleaning"] = round(time.time() - t0, 2)
            stage_timings["total"] = round(time.time() - t_start_total, 2)
            return {
                "id": dataset_id,
                "filename": display_name,
                "status": "failed",
                "stages": stages,
                "stage_timings": stage_timings,
                "error": f"Data cleaning error: {str(e)}"
            }

        # Analytical Modules
        analyzer = DataAnalyzer(cleaned_df, col_types)

        # 5. Stage: Statistics
        notify("Statistics", "running")
        t0 = time.time()
        try:
            desc_stats = analyzer.compute_descriptive_stats()
            cat_stats = analyzer.compute_categorical_analysis()
            stage_timings["statistics"] = round(time.time() - t0, 2)
            notify("Statistics", "completed", stage_timings["statistics"])
        except Exception as e:
            desc_stats = {}
            cat_stats = {}
            stage_timings["statistics"] = round(time.time() - t0, 2)

        # 6. Stage: Correlation
        notify("Correlation", "running")
        t0 = time.time()
        try:
            correlations = analyzer.compute_correlations()
            stage_timings["correlations"] = round(time.time() - t0, 2)
            notify("Correlation", "completed", stage_timings["correlations"])
        except Exception as e:
            correlations = {"has_correlation": False, "matrix": {}, "significant_pairs": []}
            stage_timings["correlations"] = round(time.time() - t0, 2)

        # 7. Stage: Anomalies
        notify("Anomalies", "running")
        t0 = time.time()
        try:
            detector = AnomalyDetector(cleaned_df, col_types)
            anomalies_data = detector.get_comprehensive_anomalies()
            outliers = analyzer.detect_outliers()
            stage_timings["anomalies"] = round(time.time() - t0, 2)
            notify("Anomalies", "completed", stage_timings["anomalies"])
        except Exception as e:
            anomalies_data = {"total_anomalies": 0, "severity_counts": {}, "column_breakdown": {}, "anomalies_list": []}
            outliers = {"total_outliers_detected": 0, "column_outliers": {}, "top_anomalies": []}
            stage_timings["anomalies"] = round(time.time() - t0, 2)

        # 8. Stage: Clustering
        notify("Clustering", "running")
        t0 = time.time()
        try:
            clustering = analyzer.compute_clustering()
            stage_timings["clustering"] = round(time.time() - t0, 2)
            notify("Clustering", "completed", stage_timings["clustering"])
        except Exception as e:
            clustering = None
            stage_timings["clustering"] = round(time.time() - t0, 2)

        # 9. Stage: Insights
        notify("Insights", "running")
        t0 = time.time()
        try:
            temporal_trends = analyzer.compute_temporal_trends()
            analysis_results = {
                "descriptive_stats": desc_stats,
                "categorical_analysis": cat_stats,
                "correlations": correlations,
                "outliers": outliers,
                "clustering": clustering,
                "temporal_trends": temporal_trends
            }
            ai_insights = InsightEngine.generate_factual_insights(
                analysis_results, health_metrics, clean_summary
            )
            stage_timings["insights"] = round(time.time() - t0, 2)
            stages["analyzed"] = True
            notify("Insights", "completed", stage_timings["insights"])
        except Exception as e:
            stage_timings["insights"] = round(time.time() - t0, 2)
            stage_timings["total"] = round(time.time() - t_start_total, 2)
            return {
                "id": dataset_id,
                "filename": display_name,
                "status": "failed",
                "stages": stages,
                "stage_timings": stage_timings,
                "error": f"Statistical insight error: {str(e)}"
            }

        total_elapsed = round(time.time() - t_start_total, 2)
        stage_timings["total"] = total_elapsed
        notify("Completed", "done", total_elapsed)

        return {
            "id": dataset_id,
            "filename": display_name,
            "raw_filename": raw_filename,
            "raw_df": raw_df,
            "cleaned_df": cleaned_df,
            "original_row_count": len(raw_df),
            "cleaned_row_count": len(cleaned_df),
            "column_count": len(cleaned_df.columns),
            "column_types": col_types,
            "missing_value_count": health_metrics.get("total_missing", 0),
            "duplicate_count": clean_summary.get("duplicates_detected", 0),
            "health_metrics": health_metrics,
            "cleaning_summary": clean_summary,
            "analysis_results": analysis_results,
            "anomalies_data": anomalies_data,
            "ai_insights": ai_insights,
            "status": "success",
            "stages": stages,
            "stage_timings": stage_timings,
            "error": None
        }

    @classmethod
    def process_batch(
        cls,
        files_list: List[Any],
        existing_datasets: Optional[Dict[str, Dict[str, Any]]] = None,
        remove_duplicates: bool = True,
        progress_callback: Optional[Callable[[int, int, str, str, str, float], None]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Processes a list of uploaded files, keeping valid existing datasets and providing live progress callbacks.
        Returns:
            (updated_datasets_dict, processing_log_list)
        """
        datasets = dict(existing_datasets) if existing_datasets else {}
        existing_names = [d["filename"] for d in datasets.values()]
        processing_logs = []
        total_files = len(files_list)

        for idx, file_item in enumerate(files_list, start=1):
            # Determine filename
            if hasattr(file_item, "name"):
                name = file_item.name
            elif isinstance(file_item, str):
                name = os.path.basename(file_item)
            else:
                name = "dataset.csv"

            res = cls.process_single_file(
                file_item,
                name,
                existing_display_names=existing_names,
                remove_duplicates=remove_duplicates,
                progress_callback=progress_callback,
                file_index=idx,
                total_files=total_files
            )

            if res["status"] == "success":
                datasets[res["id"]] = res
                existing_names.append(res["filename"])
                processing_logs.append({
                    "filename": res["filename"],
                    "status": "success",
                    "id": res["id"],
                    "stages": res["stages"],
                    "stage_timings": res.get("stage_timings", {}),
                    "message": "Loaded, Cleaned, Analyzed"
                })
            else:
                processing_logs.append({
                    "filename": res["filename"],
                    "status": "failed",
                    "id": res["id"],
                    "stages": res["stages"],
                    "stage_timings": res.get("stage_timings", {}),
                    "error": res.get("error", "Unknown error"),
                    "message": res.get("error", "Unknown error")
                })

        return datasets, processing_logs

    @staticmethod
    def get_batch_summary_table(datasets: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Produces a clean comparative overview table across all active datasets.
        Values are calculated from real datasets (no mock data).
        """
        rows = []
        for d in datasets.values():
            if d.get("status") != "success":
                continue
            h = d.get("health_metrics", {})
            anoms = d.get("anomalies_data", {})
            rows.append({
                "Dataset": d["filename"],
                "Rows": f"{d['cleaned_row_count']:,}",
                "Columns": d["column_count"],
                "Missing Values": f"{d['missing_value_count']:,}",
                "Health Score": f"{h.get('score', 100)}/100",
                "Anomalies": anoms.get("total_anomalies", 0)
            })
        return pd.DataFrame(rows)
