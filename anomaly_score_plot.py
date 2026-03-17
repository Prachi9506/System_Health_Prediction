import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("system_with_anomalies.csv")

plt.figure(figsize=(10,5))
plt.plot(df["anomaly_score"])

plt.title("Anomaly Score Over Time")
plt.xlabel("Time Index")
plt.ylabel("Anomaly Score")

plt.show()
