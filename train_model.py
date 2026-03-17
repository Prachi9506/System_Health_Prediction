import pandas as pd

# Load data
df = pd.read_csv("pc_health_data.csv")

# Drop timestamp 
df = df.drop(columns=["timestamp"])

print("Shape:", df.shape)
print("\nLabel Distribution:")
print(df["label"].value_counts())

# Separate features and label
X = df.drop("label", axis=1)
y = df["label"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
