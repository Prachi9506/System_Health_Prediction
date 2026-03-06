import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("pc_health_data.csv")

print(df.head())
print(df.shape)

# Remove non-numeric columns
df = df.drop(columns=["timestamp", "label"], errors="ignore")

print("\nColumns used for ML:")
print(df.columns)

# Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Train Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

model.fit(scaled_data)

# Predictions
df["anomaly_label"] = model.predict(scaled_data)
df["anomaly_score"] = model.decision_function(scaled_data)

# anomaly count
print("\nAnomaly Distribution:")
print(df["anomaly_label"].value_counts())

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=df["cpu_usage"],
    y=df["ram_usage"],
    hue=df["anomaly_label"],
    palette={1:"blue",-1:"red"}
)

plt.title("CPU vs RAM Anomaly Detection")
plt.show()
