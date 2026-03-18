import pandas as pd

df = pd.read_csv("pc_health_data.csv")

if "timestamp" in df.columns:
    df = df.sort_values(by="timestamp")

df["RUL"] = list(range(len(df), 0, -1))

df["failure"] = 0
df.loc[df.index[-1], "failure"] = 1

df.to_csv("rul_data.csv", index=False)

print("rul_data.csv created successfully!")
