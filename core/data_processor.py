"""
NexInsight - Core Data Processor
Handles file loading, dynamic column type detection, data cleaning,
audit trail generation, and dynamic dataset health score calculation.
"""

import io
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List


class DataProcessor:
    """Robust data processing pipeline that operates on any unseen dataset."""

    def __init__(self):
        self.raw_df: pd.DataFrame = None
        self.cleaned_df: pd.DataFrame = None
        self.column_types: Dict[str, str] = {}
        self.cleaning_summary: Dict[str, Any] = {}
        self.health_metrics: Dict[str, Any] = {}

    @staticmethod
    def load_file(file_obj, filename: str) -> pd.DataFrame:
        """
        Loads CSV or Excel files from a file path or file-like buffer.
        Automatically handles delimiters, encodings, and Excel sheets.
        """
        ext = filename.split(".")[-1].lower() if "." in filename else "csv"
        
        if ext in ["xlsx", "xls"]:
            return pd.read_excel(file_obj)
        else:
            # For CSV, attempt auto-detection of separator
            if hasattr(file_obj, "read"):
                content = file_obj.read()
                # If bytes, decode with fallback encodings
                if isinstance(content, bytes):
                    for enc in ["utf-8", "latin1", "cp1252"]:
                        try:
                            text_content = content.decode(enc)
                            break
                        except UnicodeDecodeError:
                            text_content = None
                    if text_content is None:
                        text_content = content.decode("utf-8", errors="replace")
                else:
                    text_content = content
                
                # Check header sample for delimiter
                sample = text_content[:4096]
                sep = ","
                for candidate in [",", "\t", ";", "|"]:
                    if candidate in sample:
                        sep = candidate
                        break
                return pd.read_csv(io.StringIO(text_content), sep=sep)
            else:
                # Direct filepath
                try:
                    return pd.read_csv(file_obj, sep=None, engine="python")
                except Exception:
                    return pd.read_csv(file_obj)

    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically infers column types:
        - 'Date'
        - 'Numeric'
        - 'Boolean'
        - 'Categorical'
        - 'Text'
        """
        col_types = {}
        for col in df.columns:
            series = df[col].dropna()
            
            if len(series) == 0:
                col_types[col] = "Text"
                continue

            # 1. Check Boolean
            # Direct bool dtype or binary string/number
            if pd.api.types.is_bool_dtype(df[col]):
                col_types[col] = "Boolean"
                continue
                
            unique_vals = set(series.astype(str).str.strip().str.lower().unique())
            bool_pairs = [
                {"true", "false"},
                {"yes", "no"},
                {"1", "0"},
                {"y", "n"},
                {"t", "f"}
            ]
            if any(unique_vals.issubset(pair) for pair in bool_pairs) and len(unique_vals) <= 2:
                col_types[col] = "Boolean"
                continue

            # 2. Check Date / Datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                col_types[col] = "Date"
                continue
            
            # If string, test for datetime parsing
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                sample = series.head(50).astype(str)
                # Quick check if strings resemble dates (contain separators like -, /, :)
                has_date_delims = sample.str.contains(r"[\-/:]", regex=True).mean() > 0.6
                if has_date_delims:
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                        valid_ratio = parsed.notna().mean()
                        if valid_ratio >= 0.8:
                            # Additional check: reasonable years
                            years = parsed.dropna().dt.year
                            if (years >= 1950).all() and (years <= 2100).all():
                                col_types[col] = "Date"
                                continue
                    except Exception:
                        pass

            # 3. Check Numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                # If numeric but only 2-5 unique values with small integer range, could be categorical or rating
                # If unique values < 6 and total rows > 20, check if used as category
                n_unique = series.nunique()
                if n_unique <= 4 and series.dtype in [np.int64, np.int32, int]:
                    # Let user inspect as numeric or categorical depending on context
                    col_types[col] = "Numeric"
                else:
                    col_types[col] = "Numeric"
                continue

            # If string, test if it's formatted numeric (currency like "$1,200", percentages "45%")
            if pd.api.types.is_object_dtype(df[col]):
                cleaned_num = series.astype(str).str.replace(r"[\$,€,£,%]", "", regex=True).str.replace(",", "", regex=False).str.strip()
                converted = pd.to_numeric(cleaned_num, errors="coerce")
                if converted.notna().mean() >= 0.85:
                    col_types[col] = "Numeric"
                    continue

            # 4. Check Categorical vs Text
            n_unique = series.nunique()
            total_count = len(series)
            
            # Heuristic: low cardinality or ratio of unique to total is small
            if n_unique <= 50 or (n_unique / total_count < 0.25 and n_unique < 250):
                col_types[col] = "Categorical"
            else:
                col_types[col] = "Text"

        self.column_types = col_types
        return col_types

    def clean_dataset(self, df: pd.DataFrame, remove_duplicates: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Performs data cleaning:
        - Deduplication
        - Missing value imputation based on column type
        - Normalizing string casing and stripping whitespace
        - Standardizing date columns
        - Converting cleanable numerics
        Returns (cleaned_df, cleaning_summary)
        """
        cleaned = df.copy()
        raw_rows = len(cleaned)
        details = []

        # 1. Deduplication
        duplicates_count = int(cleaned.duplicated().sum())
        if remove_duplicates and duplicates_count > 0:
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
            details.append(f"Removed {duplicates_count:,} duplicate rows.")
        elif duplicates_count > 0:
            details.append(f"Detected {duplicates_count:,} duplicate rows (retained per settings).")

        # Re-detect types on deduplicated data
        col_types = self.detect_column_types(cleaned)
        
        missing_handled = 0
        formatting_corrected = 0
        date_cols_detected = 0

        for col, ctype in col_types.items():
            null_count = int(cleaned[col].isna().sum())

            # A. Clean strings & whitespaces
            if ctype in ["Categorical", "Text", "Boolean"]:
                # Strip leading/trailing spaces
                if cleaned[col].dtype == object:
                    cleaned[col] = cleaned[col].astype(str).str.strip()
                    cleaned[col] = cleaned[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

            # B. Standardize Dates
            if ctype == "Date":
                date_cols_detected += 1
                try:
                    cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
                    # Forward-fill if missing or leave NaT
                    if null_count > 0:
                        cleaned[col] = cleaned[col].ffill().bfill()
                        missing_handled += null_count
                        details.append(f"Imputed {null_count} missing dates in '{col}' using temporal interpolation.")
                except Exception:
                    pass

            # C. Standardize Numerics
            elif ctype == "Numeric":
                if not pd.api.types.is_numeric_dtype(cleaned[col]):
                    cleaned_num = cleaned[col].astype(str).str.replace(r"[\$,€,£,%]", "", regex=True).str.replace(",", "", regex=False).str.strip()
                    cleaned[col] = pd.to_numeric(cleaned_num, errors="coerce")
                    formatting_corrected += 1
                    details.append(f"Standardized numeric formatting in '{col}'.")
                
                # Impute missing values with median
                if null_count > 0:
                    median_val = cleaned[col].median()
                    if pd.notna(median_val):
                        cleaned[col] = cleaned[col].fillna(median_val)
                        missing_handled += null_count
                        details.append(f"Imputed {null_count} missing numeric values in '{col}' with median ({median_val:.2f}).")

            # D. Standardize Categoricals
            elif ctype == "Categorical":
                # Normalize case if variations exist (e.g. 'Yes', 'yes')
                unique_vals = cleaned[col].dropna().unique()
                if len(unique_vals) > 0:
                    val_map = {}
                    seen_lower = {}
                    for v in unique_vals:
                        v_str = str(v)
                        v_lower = v_str.lower()
                        if v_lower in seen_lower:
                            val_map[v] = seen_lower[v_lower]
                            formatting_corrected += 1
                        else:
                            # Standardize to title or original
                            seen_lower[v_lower] = v_str.capitalize() if v_str.islower() else v_str
                            val_map[v] = seen_lower[v_lower]
                    if formatting_corrected > 0:
                        cleaned[col] = cleaned[col].map(lambda x: val_map.get(x, x))
                
                # Impute missing categoricals
                if null_count > 0:
                    mode_series = cleaned[col].mode()
                    mode_val = mode_series[0] if len(mode_series) > 0 else "Unspecified"
                    cleaned[col] = cleaned[col].fillna(mode_val)
                    missing_handled += null_count
                    details.append(f"Imputed {null_count} missing values in '{col}' with mode ('{mode_val}').")

            # E. Booleans
            elif ctype == "Boolean":
                mapping = {"true": True, "1": True, "yes": True, "y": True, "t": True,
                           "false": False, "0": False, "no": False, "n": False, "f": False}
                cleaned[col] = cleaned[col].astype(str).str.lower().map(lambda x: mapping.get(x, np.nan))
                if null_count > 0:
                    cleaned[col] = cleaned[col].fillna(False)
                    missing_handled += null_count

            # F. Text
            elif ctype == "Text":
                if null_count > 0:
                    cleaned[col] = cleaned[col].fillna("Unknown")
                    missing_handled += null_count

        summary = {
            "raw_rows": raw_rows,
            "cleaned_rows": len(cleaned),
            "duplicates_removed": duplicates_count if remove_duplicates else 0,
            "duplicates_detected": duplicates_count,
            "missing_values_handled": missing_handled,
            "formatting_inconsistencies_corrected": formatting_corrected,
            "date_columns_detected": date_cols_detected,
            "details": details
        }
        self.cleaned_df = cleaned
        self.cleaning_summary = summary
        return cleaned, summary

    def calculate_health_score(self, df_raw: pd.DataFrame, df_cleaned: pd.DataFrame) -> Dict[str, Any]:
        """
        Dynamically calculates a comprehensive dataset health score (0 - 100).
        Factors:
        - Missing cell ratio (up to 35% penalty)
        - Duplicate rows ratio (up to 25% penalty)
        - Column completeness and zero-variance columns (up to 20% penalty)
        - Type consistency (up to 20% penalty)
        """
        total_cells = df_raw.shape[0] * df_raw.shape[1]
        if total_cells == 0:
            return {"score": 0, "status": "Empty Dataset", "breakdown": {}}

        total_missing = int(df_raw.isna().sum().sum())
        missing_ratio = total_missing / total_cells
        missing_penalty = min(35.0, missing_ratio * 100.0 * 1.5)

        total_rows = len(df_raw)
        duplicate_rows = int(df_raw.duplicated().sum())
        dup_ratio = duplicate_rows / max(1, total_rows)
        dup_penalty = min(25.0, dup_ratio * 100.0 * 2.0)

        # Single value columns (zero variance / useless columns)
        single_val_cols = sum(1 for col in df_raw.columns if df_raw[col].nunique(dropna=False) <= 1)
        single_val_penalty = min(20.0, (single_val_cols / max(1, len(df_raw.columns))) * 40.0)

        # High cardinality text penalty or dirty mixed values
        consistency_penalty = 0.0
        for col in df_raw.columns:
            types_in_col = df_raw[col].dropna().apply(lambda x: type(x).__name__).nunique()
            if types_in_col > 2:
                consistency_penalty += 3.0
        consistency_penalty = min(20.0, consistency_penalty)

        total_penalty = missing_penalty + dup_penalty + single_val_penalty + consistency_penalty
        score = max(0, min(100, int(round(100.0 - total_penalty))))

        if score >= 90:
            status = "Excellent"
            status_color = "#10B981"
        elif score >= 75:
            status = "Good"
            status_color = "#3B82F6"
        elif score >= 60:
            status = "Fair"
            status_color = "#F59E0B"
        else:
            status = "Needs Attention"
            status_color = "#EF4444"

        completeness_pct = round((1.0 - missing_ratio) * 100, 1)
        uniqueness_pct = round((1.0 - dup_ratio) * 100, 1)
        validity_pct = round(max(0, 100.0 - single_val_penalty - consistency_penalty), 1)

        health_data = {
            "score": score,
            "status": status,
            "status_color": status_color,
            "total_rows": total_rows,
            "total_columns": df_raw.shape[1],
            "total_missing": total_missing,
            "duplicate_rows": duplicate_rows,
            "completeness_pct": completeness_pct,
            "uniqueness_pct": uniqueness_pct,
            "validity_pct": validity_pct,
            "breakdown": {
                "Completeness": f"{completeness_pct}%",
                "Uniqueness": f"{uniqueness_pct}%",
                "Structure Validity": f"{validity_pct}%",
                "Single-Value Columns": single_val_cols
            }
        }
        self.health_metrics = health_data
        return health_data

    def process(self, file_obj, filename: str, remove_duplicates: bool = True) -> Dict[str, Any]:
        """
        Executes the full automated ingestion and cleaning pipeline on any file.
        """
        raw = self.load_file(file_obj, filename)
        self.raw_df = raw
        
        # Calculate health metrics before cleaning
        cleaned, cleaning_summary = self.clean_dataset(raw, remove_duplicates=remove_duplicates)
        health_metrics = self.calculate_health_score(raw, cleaned)
        col_types = self.detect_column_types(cleaned)

        return {
            "filename": filename,
            "raw_df": raw,
            "cleaned_df": cleaned,
            "column_types": col_types,
            "cleaning_summary": cleaning_summary,
            "health_metrics": health_metrics
        }
