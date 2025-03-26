import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("iot_telemetry_data.csv")

# Split data into features and labels
X = df[["packet_drops", "retry_counts", "message_delivery_time"]]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("XGBoost Model Performance:")
print(classification_report(y_test, y_pred))
