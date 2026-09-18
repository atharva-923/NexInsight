import pandas as pd
from typing import Dict, Any, List
from core.anomalies import AnomalyDetector

class SafeQueryExecutor:
    """
    Executes a structured JSON pipeline against a Pandas DataFrame safely.
    Strictly forbids eval(), exec(), and df.query(). Uses explicit boolean masks.
    """

    ALLOWED_OPERATORS = ["==", "!=", ">", "<", ">=", "<=", "in", "not_in"]
    ALLOWED_METRICS = ["sum", "mean", "median", "count", "min", "max", "nunique"]

    def __init__(self, df: pd.DataFrame, column_types: Dict[str, str]):
        self.df = df
        self.column_types = column_types

    def _validate_columns(self, current_df: pd.DataFrame, cols: List[str]):
        """Ensure all columns exist in the current DataFrame."""
        for col in cols:
            if col not in current_df.columns:
                raise ValueError(f"Security constraint violated: Unknown column '{col}'")

    def execute_pipeline(self, pipeline: List[Dict[str, Any]]) -> pd.DataFrame:
        """Executes the pipeline steps iteratively."""
        if len(pipeline) > 8:
            raise ValueError("Pipeline exceeds maximum length of 8 steps.")

        current_df = self.df.copy()

        for step in pipeline:
            op = step.get("operation")
            
            if op == "filter":
                current_df = self._apply_filter(current_df, step)
            elif op == "group_aggregate":
                current_df = self._apply_group_aggregate(current_df, step)
            elif op == "aggregate":
                current_df = self._apply_aggregate(current_df, step)
            elif op == "sort":
                current_df = self._apply_sort(current_df, step)
            elif op == "limit":
                current_df = self._apply_limit(current_df, step)
            elif op == "calculate_anomaly":
                current_df = self._apply_calculate_anomaly(current_df, step)
            else:
                raise ValueError(f"Security constraint violated: Unknown operation '{op}'")

        return current_df

    def _apply_filter(self, df: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
        col = step.get("column")
        operator = step.get("operator")
        val = step.get("value")

        if not col or not operator or val is None:
            raise ValueError("Filter step missing required fields.")
        if operator not in self.ALLOWED_OPERATORS:
            raise ValueError(f"Security constraint violated: Unknown operator '{operator}'")
        self._validate_columns(df, [col])

        # Explicit boolean masking to prevent code execution
        if operator == "==":
            return df[df[col] == val]
        elif operator == "!=":
            return df[df[col] != val]
        elif operator == ">":
            return df[df[col] > float(val)]
        elif operator == "<":
            return df[df[col] < float(val)]
        elif operator == ">=":
            return df[df[col] >= float(val)]
        elif operator == "<=":
            return df[df[col] <= float(val)]
        elif operator == "in":
            if not isinstance(val, list):
                raise ValueError("'in' operator requires a list value.")
            return df[df[col].isin(val)]
        elif operator == "not_in":
            if not isinstance(val, list):
                raise ValueError("'not_in' operator requires a list value.")
            return df[df[col].isin(val) == False]
        return df

    def _apply_group_aggregate(self, df: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
        group_by = step.get("group_by")
        if not group_by or not isinstance(group_by, list):
            raise ValueError("Invalid schema: 'group_aggregate' requires a non-empty group_by list.")
            
        if len(group_by) > 3:
            raise ValueError("Security constraint violated: group_by exceeds 3 columns limit.")
            
        self._validate_columns(df, group_by)

        aggregations = step.get("aggregations")
        if not aggregations or not isinstance(aggregations, list):
            raise ValueError("Invalid schema: 'group_aggregate' requires a non-empty aggregations list.")
            
        agg_dict = {}
        rename_dict = {}

        for agg in aggregations:
            if not isinstance(agg, dict):
                raise ValueError("Invalid schema: Aggregation must be a dictionary.")
            col = agg.get("column")
            metric = agg.get("metric")
            if not col or not metric:
                raise ValueError("Invalid schema: Aggregation requires column and metric.")
                
            out_name = agg.get("output_name")
            if out_name is not None and (not isinstance(out_name, str) or not out_name.strip()):
                raise ValueError("Invalid schema: output_name must be a non-empty string if provided.")
            if not out_name:
                out_name = f"{col}_{metric}"
            
            if metric not in self.ALLOWED_METRICS:
                raise ValueError(f"Security constraint violated: Unknown metric '{metric}'")
            self._validate_columns(df, [col])

            if col not in agg_dict:
                agg_dict[col] = []
            
            # Map common metric strings to pandas functions/strings
            # count and nunique don't require numeric, others generally do but pandas handles it
            agg_dict[col].append(metric)
            rename_dict[f"{col}_{metric}"] = out_name

        if not agg_dict:
            raise ValueError("No aggregations specified.")

        grouped = df.groupby(group_by).agg(agg_dict)
        
        # Flatten MultiIndex columns
        grouped.columns = [f"{col}_{met}" for col, met in grouped.columns]
        grouped = grouped.reset_index()
        
        # Rename columns based on output_name mapping
        grouped = grouped.rename(columns=rename_dict)
        return grouped

    def _apply_aggregate(self, df: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
        aggregations = step.get("aggregations")
        if not aggregations or not isinstance(aggregations, list):
            raise ValueError("Invalid schema: 'aggregate' requires a non-empty aggregations list.")
            
        result = {}

        for agg in aggregations:
            if not isinstance(agg, dict):
                raise ValueError("Invalid schema: Aggregation must be a dictionary.")
            col = agg.get("column")
            metric = agg.get("metric")
            if not col or not metric:
                raise ValueError("Invalid schema: Aggregation requires column and metric.")
                
            out_name = agg.get("output_name")
            if out_name is not None and (not isinstance(out_name, str) or not out_name.strip()):
                raise ValueError("Invalid schema: output_name must be a non-empty string if provided.")
            if not out_name:
                out_name = f"{col}_{metric}"
            
            if metric not in self.ALLOWED_METRICS:
                raise ValueError(f"Security constraint violated: Unknown metric '{metric}'")
            self._validate_columns(df, [col])
            
            series = df[col]
            if metric == "sum": val = series.sum()
            elif metric == "mean": val = series.mean()
            elif metric == "median": val = series.median()
            elif metric == "count": val = series.count()
            elif metric == "min": val = series.min()
            elif metric == "max": val = series.max()
            elif metric == "nunique": val = series.nunique()
            else: val = None
            
            result[out_name] = [val]

        return pd.DataFrame(result)

    def _apply_sort(self, df: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
        col = step.get("column")
        order = step.get("order", "ascending")
        
        if not col:
            raise ValueError("Sort missing column.")
        if order not in ["ascending", "descending"]:
            raise ValueError(f"Invalid schema: Sort order must be 'ascending' or 'descending', got '{order}'.")
            
        self._validate_columns(df, [col])
        
        ascending = order == "ascending"
        return df.sort_values(by=col, ascending=ascending)

    def _apply_limit(self, df: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
        val = step.get("value")
        if not isinstance(val, int) or val < 1:
            raise ValueError("Limit requires a positive integer value.")
        return df.head(val)

    def _apply_calculate_anomaly(self, df: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
        """
        Uses the existing AnomalyDetector to calculate anomalies on the current DataFrame state.
        Returns a DataFrame of anomalous rows for further grouping or filtering.
        """
        detector = AnomalyDetector(df, self.column_types)
        res = detector.get_comprehensive_anomalies()
        anomalies_list = res.get("anomalies_list", [])
        
        if not anomalies_list:
            # Return empty DataFrame with expected columns if no anomalies
            return pd.DataFrame(columns=["row_index", "column", "value", "normal_range", "deviation", "z_score", "severity", "reason"])
            
        anom_df = pd.DataFrame(anomalies_list)
        return anom_df
