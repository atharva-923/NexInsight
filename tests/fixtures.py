"""
NexInsight Test Fixtures
Generates synthetic datasets in-process so no committed CSV/XLSX files are required.
Each generator matches the exact schema and intentional characteristics (outliers,
missing values, duplicates) that tests were written to verify.
"""

import os
import numpy as np
import pandas as pd


def make_dataset_a(n: int = 500) -> pd.DataFrame:
    """
    Sales dataset: numeric + categorical + dates, zero missing values, with outliers.
    Columns: Transaction_Date, Region, Product, Units_Sold, Unit_Price,
             Marketing_Spend, Customer_Rating, Total_Revenue
    """
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=120, freq="D")
    regions = ["North America", "Europe", "Asia Pacific", "Latin America"]
    products = ["Enterprise Suite", "Cloud Storage", "AI Assistant", "Security Shield"]
    df = pd.DataFrame({
        "Transaction_Date": np.random.choice(dates.astype(str), n),
        "Region": np.random.choice(regions, n, p=[0.4, 0.3, 0.2, 0.1]),
        "Product": np.random.choice(products, n),
        "Units_Sold": np.random.randint(1, 80, n),
        "Unit_Price": np.random.uniform(50.0, 950.0, n).round(2),
        "Marketing_Spend": np.random.uniform(100.0, 2500.0, n).round(2),
        "Customer_Rating": np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.25, 0.4, 0.2]),
    })
    df["Total_Revenue"] = (df["Units_Sold"] * df["Unit_Price"]).round(2)
    df.loc[12, "Total_Revenue"] = 185000.0  # intentional outlier
    df.loc[45, "Units_Sold"] = 450           # intentional outlier
    return df


def make_dataset_b(n: int = 600) -> pd.DataFrame:
    """
    Server metrics: numeric + categorical + timestamps, with outliers.
    Columns: Timestamp, Node_Identifier, Cluster_Zone, CPU_Load_Pct,
             Memory_Used_GB, Network_Throughput_MBps, Active_Threads, Health_Flag
    """
    np.random.seed(43)
    timestamps = pd.date_range(start="2025-06-01", periods=600, freq="5min")
    servers = [f"srv-node-0{i}" for i in range(1, 9)]
    clusters = ["Cluster-East", "Cluster-West", "Cluster-Central"]
    df = pd.DataFrame({
        "Timestamp": np.random.choice(timestamps.astype(str), n),
        "Node_Identifier": np.random.choice(servers, n),
        "Cluster_Zone": np.random.choice(clusters, n),
        "CPU_Load_Pct": np.random.normal(55, 15, n).clip(5, 100).round(2),
        "Memory_Used_GB": np.random.normal(32, 8, n).clip(4, 64).round(2),
        "Network_Throughput_MBps": np.random.exponential(120, n).round(2),
        "Active_Threads": np.random.randint(20, 300, n),
        "Health_Flag": np.random.choice(
            ["HEALTHY", "WARNING", "CRITICAL"], n, p=[0.85, 0.12, 0.03]
        ),
    })
    df.loc[7, "CPU_Load_Pct"] = 99.8    # intentional outlier
    df.loc[99, "Active_Threads"] = 1850  # intentional outlier
    return df


def make_dataset_c(n: int = 400, n_dups: int = 35) -> pd.DataFrame:
    """
    Survey data: intentional missing values + duplicate rows (dirty dataset).
    Columns: Respondent_ID, Department, Tenure_Years, Salary_Band,
             Satisfaction_Score, Feedback_Sentiment, Promote_Eligible
    """
    np.random.seed(44)
    respondents = [f"RESP-{1000 + i}" for i in range(n)]
    df = pd.DataFrame({
        "Respondent_ID": respondents,
        "Department": np.random.choice(
            ["Engineering", "Marketing", "Sales", "Support", "Finance", None],
            n, p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1]
        ),
        "Tenure_Years": np.random.choice(
            [1.0, 2.5, 4.0, 6.5, np.nan, 10.0, 12.0], n
        ),
        "Salary_Band": np.random.choice(
            ["Tier 1", "Tier 2", "Tier 3", "Tier 4", None], n
        ),
        "Satisfaction_Score": np.random.choice(
            [1.0, 2.0, 3.0, 4.0, 5.0, np.nan], n
        ),
        "Feedback_Sentiment": np.random.choice(
            ["Positive", "positive", "POSITIVE", "Neutral", "Negative", None], n
        ),
        "Promote_Eligible": np.random.choice(
            ["Yes", "No", "yes", "no", None], n
        ),
    })
    dups = df.sample(n_dups, random_state=42)
    df = pd.concat([df, dups], ignore_index=True)
    return df


def make_dataset_d(n: int = 450) -> pd.DataFrame:
    """
    Categorical-only data (no numerics, no dates).
    Columns: Account_Tier, Primary_Industry, Payment_Method,
             Support_Level, Contract_Renewal_Term, Churn_Risk_Status, NPS_Category
    """
    np.random.seed(45)
    tiers = ["Free", "Standard", "Premium", "Enterprise"]
    industries = ["FinTech", "Healthcare", "Education", "E-commerce", "Manufacturing", "Media"]
    df = pd.DataFrame({
        "Account_Tier": np.random.choice(tiers, n, p=[0.4, 0.3, 0.2, 0.1]),
        "Primary_Industry": np.random.choice(industries, n),
        "Payment_Method": np.random.choice(
            ["Credit Card", "Wire Transfer", "PayPal", "ACH"], n
        ),
        "Support_Level": np.random.choice(
            ["Basic", "Priority", "Dedicated Manager"], n
        ),
        "Contract_Renewal_Term": np.random.choice(
            ["Monthly", "Annual", "Multi-Year"], n
        ),
        "Churn_Risk_Status": np.random.choice(
            ["Low Risk", "Medium Risk", "High Risk"], n, p=[0.7, 0.2, 0.1]
        ),
        "NPS_Category": np.random.choice(
            ["Promoter", "Passive", "Detractor"], n, p=[0.55, 0.3, 0.15]
        ),
    })
    return df


def make_dataset_e(n: int = 500) -> pd.DataFrame:
    """
    Numeric-only sensor data (no categoricals, no dates) with outliers.
    Columns: Sensor_Alpha_Voltage, Sensor_Beta_Current, Core_Temperature_C,
             Vibration_Frequency_Hz, Hydraulic_Pressure_PSI,
             Acoustic_Emission_dB, Efficiency_Ratio
    """
    np.random.seed(46)
    df = pd.DataFrame({
        "Sensor_Alpha_Voltage": np.random.normal(12.0, 0.8, n).round(3),
        "Sensor_Beta_Current": np.random.normal(4.5, 0.4, n).round(3),
        "Core_Temperature_C": np.random.normal(68.5, 5.2, n).round(2),
        "Vibration_Frequency_Hz": np.random.normal(120.0, 15.0, n).round(1),
        "Hydraulic_Pressure_PSI": np.random.normal(2100.0, 180.0, n).round(1),
        "Acoustic_Emission_dB": np.random.normal(42.0, 6.0, n).round(2),
        "Efficiency_Ratio": np.random.uniform(0.75, 0.98, n).round(4),
    })
    df.loc[25, "Core_Temperature_C"] = 115.0      # intentional outlier
    df.loc[70, "Hydraulic_Pressure_PSI"] = 3800.0  # intentional outlier
    return df


def make_batch_xlsx(n: int = 100) -> pd.DataFrame:
    """Simple n-row DataFrame for XLSX batch upload tests."""
    np.random.seed(99)
    return pd.DataFrame({
        "ID": range(1, n + 1),
        "Value": np.random.uniform(10.0, 500.0, n).round(2),
        "Category": np.random.choice(["Alpha", "Beta", "Gamma"], n),
    })


def write_csv(df: pd.DataFrame, path: str) -> None:
    """Writes a DataFrame to a CSV file at the given path."""
    df.to_csv(path, index=False)


def write_xlsx(df: pd.DataFrame, path: str) -> None:
    """Writes a DataFrame to an XLSX file at the given path."""
    df.to_excel(path, index=False)


def write_corrupt_xlsx(path: str) -> None:
    """Writes invalid bytes to path to trigger an XLSX parse failure."""
    with open(path, "wb") as f:
        f.write(b"NOTANXLSX_CORRUPT_BINARY_DATA_!@#$%^&*()")


def write_empty_csv(path: str) -> None:
    """Writes a zero-byte file to trigger an empty-dataset failure."""
    with open(path, "wb") as f:
        pass
