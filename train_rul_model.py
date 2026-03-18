import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("rul_data.csv")

df = df.drop(columns=["timestamp", "failure"], errors="ignore")

X = df.drop(columns=["RUL", "label"], errors="ignore")
y = df["RUL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))

joblib.dump(model, "rul_model.pkl")
joblib.dump(scaler, "rul_scaler.pkl")

print("RUL model saved successfully!")
