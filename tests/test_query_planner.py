import pytest
import pandas as pd
from unittest.mock import patch
from core.query_planner import QueryPlanner
from core.safe_executor import SafeQueryExecutor
from core.qa_engine import DataQAEngine

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "region": ["North", "South", "East", "West", "North", "South"],
        "category": ["A", "A", "B", "B", "A", "C"],
        "sales": [100, 200, 150, 300, 400, 50],
        "anomalies_flag": [0, 0, 0, 1, 0, 0] # Example for checking
    })

@pytest.fixture
def qa_engine(sample_df):
    column_types = {"region": "Categorical", "category": "Categorical", "sales": "Numeric", "anomalies_flag": "Numeric"}
    return DataQAEngine(
        df=sample_df,
        column_types=column_types,
        analysis_results={"outliers": {
            "total_outliers_detected": 1,
            "top_anomalies": [{"column": "sales", "row_index": 3, "value": 1000, "normal_range": "0-100"}]
        }},
        health_metrics={"score": 100}
    )

def test_safe_executor_filtering(sample_df):
    executor = SafeQueryExecutor(sample_df, {})
    pipeline = [{"operation": "filter", "column": "region", "operator": "==", "value": "North"}]
    res = executor.execute_pipeline(pipeline)
    assert len(res) == 2
    assert res["sales"].sum() == 500

def test_safe_executor_grouped_aggregation(sample_df):
    executor = SafeQueryExecutor(sample_df, {})
    pipeline = [{
        "operation": "group_aggregate",
        "group_by": ["category"],
        "aggregations": [{"column": "sales", "metric": "sum", "output_name": "total_sales"}]
    }]
    res = executor.execute_pipeline(pipeline)
    assert len(res) == 3
    assert res[res["category"] == "A"]["total_sales"].iloc[0] == 700

def test_safe_executor_sorting_and_limit(sample_df):
    executor = SafeQueryExecutor(sample_df, {})
    pipeline = [
        {"operation": "sort", "column": "sales", "order": "descending"},
        {"operation": "limit", "value": 2}
    ]
    res = executor.execute_pipeline(pipeline)
    assert len(res) == 2
    assert res.iloc[0]["sales"] == 400
    assert res.iloc[1]["sales"] == 300

def test_safe_executor_multi_step(sample_df):
    executor = SafeQueryExecutor(sample_df, {})
    pipeline = [
        {"operation": "filter", "column": "category", "operator": "==", "value": "A"},
        {"operation": "group_aggregate", "group_by": ["region"], "aggregations": [{"column": "sales", "metric": "sum", "output_name": "total_sales"}]},
        {"operation": "sort", "column": "total_sales", "order": "descending"},
        {"operation": "limit", "value": 1}
    ]
    res = executor.execute_pipeline(pipeline)
    assert len(res) == 1
    assert res.iloc[0]["region"] == "North"
    assert res.iloc[0]["total_sales"] == 500

def test_safe_executor_invalid_column(sample_df):
    executor = SafeQueryExecutor(sample_df, {})
    pipeline = [{"operation": "filter", "column": "hacked_col", "operator": "==", "value": "North"}]
    with pytest.raises(ValueError, match="Security constraint violated: Unknown column 'hacked_col'"):
        executor.execute_pipeline(pipeline)

def test_safe_executor_invalid_operation(sample_df):
    executor = SafeQueryExecutor(sample_df, {})
    pipeline = [{"operation": "drop_table", "column": "region", "operator": "==", "value": "North"}]
    with pytest.raises(ValueError, match="Security constraint violated: Unknown operation 'drop_table'"):
        executor.execute_pipeline(pipeline)

def test_safe_executor_invalid_operator(sample_df):
    executor = SafeQueryExecutor(sample_df, {})
    pipeline = [{"operation": "filter", "column": "region", "operator": "drop", "value": "North"}]
    with pytest.raises(ValueError, match="Security constraint violated: Unknown operator 'drop'"):
        executor.execute_pipeline(pipeline)

def test_safe_executor_pipeline_limit_rejected(sample_df):
    executor = SafeQueryExecutor(sample_df, {})
    pipeline = [{"operation": "limit", "value": 1}] * 9
    with pytest.raises(ValueError, match="Pipeline exceeds maximum length of 8 steps."):
        executor.execute_pipeline(pipeline)

def test_safe_executor_groupby_limit_rejected(sample_df):
    executor = SafeQueryExecutor(sample_df, {})
    pipeline = [{"operation": "group_aggregate", "group_by": ["c1", "c2", "c3", "c4"], "aggregations": []}]
    with pytest.raises(ValueError, match="Security constraint violated: group_by exceeds 3 columns limit."):
        executor.execute_pipeline(pipeline)

def test_malicious_filter_value_cannot_execute_code(sample_df):
    executor = SafeQueryExecutor(sample_df, {})
    # Injecting eval payload inside value
    malicious_val = "1; __import__('os').system('echo hacked')"
    pipeline = [{"operation": "filter", "column": "region", "operator": "==", "value": malicious_val}]
    # Since it uses boolean masking `df[df['region'] == malicious_val]`, it will just safely return empty df without executing
    res = executor.execute_pipeline(pipeline)
    assert res.empty

def test_df_query_payload_rejected(sample_df):
    executor = SafeQueryExecutor(sample_df, {})
    # If LLM hallucinates an operation trying to run df.query()
    pipeline = [{"operation": "query", "expr": "region == 'North' and @__import__('os')"}]
    with pytest.raises(ValueError, match="Security constraint violated: Unknown operation 'query'"):
        executor.execute_pipeline(pipeline)

@patch("core.llm_client.LLMClient.generate_chat_completion")
def test_malicious_planner_output_rejected(mock_gen, sample_df):
    mock_gen.return_value = {"success": True, "content": '{"pipeline": [{"operation": "eval", "code": "exit()"}]}'}
    resp = QueryPlanner.generate_plan("Do something bad", list(sample_df.columns))
    assert resp["success"] == True
    
    # SafeQueryExecutor catches it
    executor = SafeQueryExecutor(sample_df, {})
    with pytest.raises(ValueError, match="Unknown operation 'eval'"):
        executor.execute_pipeline(resp["plan"]["pipeline"])

@patch("core.anomalies.AnomalyDetector.get_comprehensive_anomalies")
def test_anomaly_grouping_uses_existing_logic(mock_anomalies, sample_df):
    # Mock the return of the anomaly engine
    mock_anomalies.return_value = {
        "anomalies_list": [
            {"row_index": 0, "column": "sales", "value": 1000, "normal_range": "0-200", "deviation": 800, "z_score": 5, "severity": "Critical", "reason": "High"}
        ]
    }
    
    executor = SafeQueryExecutor(sample_df, {"sales": "Numeric"})
    pipeline = [{"operation": "calculate_anomaly"}]
    res = executor.execute_pipeline(pipeline)
    
    assert len(res) == 1
    assert res.iloc[0]["value"] == 1000
    mock_anomalies.assert_called_once()

@patch("core.query_planner.QueryPlanner.generate_plan")
@patch("core.llm_client.LLMClient.generate_chat_completion")
def test_complex_query_routing_and_ground_truth(mock_llm, mock_planner, qa_engine):
    # Setup mock to simulate a successful plan and RAG response
    mock_planner.return_value = {
        "success": True, 
        "plan": {"pipeline": [{"operation": "limit", "value": 1}]}
    }
    mock_llm.return_value = {"success": True, "content": "The top result is here.", "model": "mock"}
    
    with patch("core.llm_client.LLMClient.is_configured", return_value=True):
        res = qa_engine.answer_query("compare the top region and filter anomalies")
        
        # It should route to complex path since "compare" and "filter" are in the query
        assert mock_planner.called
        # RAG should have received the "VERIFIED PYTHON GROUND TRUTH"
        # The fallback_message should be None, implying success
        assert res["fallback_message"] is None
        assert res["metric_highlight"] == "Complex Analysis Executed"

@patch("core.query_planner.QueryPlanner.generate_plan")
@patch("core.llm_client.LLMClient.generate_chat_completion")
def test_planner_api_failure_does_not_crash(mock_llm, mock_planner, qa_engine):
    # Simulate API failure during planning
    mock_planner.return_value = {"success": False, "error": "API Timeout"}
    mock_llm.return_value = {"success": True, "content": "mocked", "model": "mock"}
    
    with patch("core.llm_client.LLMClient.is_configured", return_value=True):
        # Query Engine should seamlessly fallback to deterministic compute
        res = qa_engine.answer_query("compare the top region and filter anomalies")
        assert res["metric_highlight"] != "Complex Analysis Executed"
        assert res["engine"] != "fallback" # Because LLM fallback only applies if final LLM fails, here planner failed and it fell back to deterministic QA which still uses RAG. Wait, actually if deterministic is used, it still sends it to RAG!
        # Thus the final answer uses Groq.

def test_simple_question_uses_deterministic_fast_path(qa_engine):
    # "What is the highest sales" is simple enough
    with patch("core.query_planner.QueryPlanner.generate_plan") as mock_planner:
        with patch("core.llm_client.LLMClient.is_configured", return_value=True):
            with patch("core.llm_client.LLMClient.generate_chat_completion", return_value={"success": True, "content": "Highest is 400", "model": "mock"}):
                qa_engine.answer_query("What is the highest sales?")
                # Planner should NOT be called for simple queries
                assert not mock_planner.called
