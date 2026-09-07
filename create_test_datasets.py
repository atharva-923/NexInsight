import os
import pandas as pd
import numpy as np

os.makedirs("test_datasets", exist_ok=True)

# Dataset A: Numerical + Categorical + Date (Sales & Marketing)
np.random.seed(42)
dates_a = pd.date_range(start="2025-01-01", periods=120, freq="D")
regions = ["North America", "Europe", "Asia Pacific", "Latin America"]
products = ["Enterprise Suite", "Cloud Storage", "AI Assistant", "Security Shield"]
data_a = {
    "Transaction_Date": np.random.choice(dates_a, 500),
    "Region": np.random.choice(regions, 500, p=[0.4, 0.3, 0.2, 0.1]),
    "Product": np.random.choice(products, 500),
    "Units_Sold": np.random.randint(1, 80, 500),
    "Unit_Price": np.random.uniform(50.0, 950.0, 500).round(2),
    "Marketing_Spend": np.random.uniform(100.0, 2500.0, 500).round(2),
    "Customer_Rating": np.random.choice([1, 2, 3, 4, 5], 500, p=[0.05, 0.1, 0.25, 0.4, 0.2])
}
df_a = pd.DataFrame(data_a)
df_a["Total_Revenue"] = (df_a["Units_Sold"] * df_a["Unit_Price"]).round(2)
# Add a few outliers
df_a.loc[12, "Total_Revenue"] = 185000.0
df_a.loc[45, "Units_Sold"] = 450
df_a.to_csv("test_datasets/dataset_a_sales.csv", index=False)
print("Dataset A created:", df_a.shape)

# Dataset B: Different names & structure (Infrastructure Server Telemetry)
timestamps = pd.date_range(start="2025-06-01 00:00:00", periods=600, freq="5min")
servers = [f"srv-node-0{i}" for i in range(1, 9)]
clusters = ["Cluster-East", "Cluster-West", "Cluster-Central"]
data_b = {
    "Timestamp": np.random.choice(timestamps, 600),
    "Node_Identifier": np.random.choice(servers, 600),
    "Cluster_Zone": np.random.choice(clusters, 600),
    "CPU_Load_Pct": np.random.normal(55, 15, 600).clip(5, 100).round(2),
    "Memory_Used_GB": np.random.normal(32, 8, 600).clip(4, 64).round(2),
    "Network_Throughput_MBps": np.random.exponential(120, 600).round(2),
    "Active_Threads": np.random.randint(20, 300, 600),
    "Health_Flag": np.random.choice(["HEALTHY", "WARNING", "CRITICAL"], 600, p=[0.85, 0.12, 0.03])
}
df_b = pd.DataFrame(data_b)
df_b.loc[7, "CPU_Load_Pct"] = 99.8
df_b.loc[99, "Active_Threads"] = 1850
df_b.to_csv("test_datasets/dataset_b_server_metrics.csv", index=False)
print("Dataset B created:", df_b.shape)

# Dataset C: Missing values + duplicate rows (Dirty Survey Data)
n_c = 400
respondents = [f"RESP-{1000 + i}" for i in range(n_c)]
data_c = {
    "Respondent_ID": respondents,
    "Department": np.random.choice(["Engineering", "Marketing", "Sales", "Support", "Finance", None], n_c, p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1]),
    "Tenure_Years": np.random.choice([1.0, 2.5, 4.0, 6.5, np.nan, 10.0, 12.0], n_c),
    "Salary_Band": np.random.choice(["Tier 1", "Tier 2", "Tier 3", "Tier 4", None], n_c),
    "Satisfaction_Score": np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, np.nan], n_c),
    "Feedback_Sentiment": np.random.choice(["Positive", "positive", "POSITIVE", "Neutral", "Negative", None], n_c),
    "Promote_Eligible": np.random.choice(["Yes", "No", "yes", "no", None], n_c)
}
df_c = pd.DataFrame(data_c)
# Add 35 duplicate rows
duplicates = df_c.sample(35, random_state=42)
df_c = pd.concat([df_c, duplicates], ignore_index=True)
df_c.to_csv("test_datasets/dataset_c_survey_dirty.csv", index=False)
print("Dataset C created:", df_c.shape)

# Dataset D: Mostly Categorical Data (Customer Segments & Preferences)
n_d = 450
tiers = ["Free", "Standard", "Premium", "Enterprise"]
industries = ["FinTech", "Healthcare", "Education", "E-commerce", "Manufacturing", "Media"]
data_d = {
    "Account_Tier": np.random.choice(tiers, n_d, p=[0.4, 0.3, 0.2, 0.1]),
    "Primary_Industry": np.random.choice(industries, n_d),
    "Payment_Method": np.random.choice(["Credit Card", "Wire Transfer", "PayPal", "ACH"], n_d),
    "Support_Level": np.random.choice(["Basic", "Priority", "Dedicated Manager"], n_d),
    "Contract_Renewal_Term": np.random.choice(["Monthly", "Annual", "Multi-Year"], n_d),
    "Churn_Risk_Status": np.random.choice(["Low Risk", "Medium Risk", "High Risk"], n_d, p=[0.7, 0.2, 0.1]),
    "NPS_Category": np.random.choice(["Promoter", "Passive", "Detractor"], n_d, p=[0.55, 0.3, 0.15])
}
df_d = pd.DataFrame(data_d)
df_d.to_csv("test_datasets/dataset_d_categorical.csv", index=False)
print("Dataset D created:", df_d.shape)

# Dataset E: Mostly Numerical Data (Industrial Sensor Telemetry)
n_e = 500
data_e = {
    "Sensor_Alpha_Voltage": np.random.normal(12.0, 0.8, n_e).round(3),
    "Sensor_Beta_Current": np.random.normal(4.5, 0.4, n_e).round(3),
    "Core_Temperature_C": np.random.normal(68.5, 5.2, n_e).round(2),
    "Vibration_Frequency_Hz": np.random.normal(120.0, 15.0, n_e).round(1),
    "Hydraulic_Pressure_PSI": np.random.normal(2100.0, 180.0, n_e).round(1),
    "Acoustic_Emission_dB": np.random.normal(42.0, 6.0, n_e).round(2),
    "Efficiency_Ratio": np.random.uniform(0.75, 0.98, n_e).round(4)
}
df_e = pd.DataFrame(data_e)
# Add some outliers
df_e.loc[25, "Core_Temperature_C"] = 115.0
df_e.loc[70, "Hydraulic_Pressure_PSI"] = 3800.0
df_e.to_csv("test_datasets/dataset_e_sensor_numeric.csv", index=False)
print("Dataset E created:", df_e.shape)
