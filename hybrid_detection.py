import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("iot_telemetry_data.csv")

# Step 1: Rule-Based Detection
# Define thresholds
# Define thresholds
# Adjusted thresholds (more sensitive)
thresholds = {
    "packet_drops": df["packet_drops"].mean() + 2 * df["packet_drops"].std(),
    "retry_counts": df["retry_counts"].mean() + 2 * df["retry_counts"].std(),
    "message_delivery_time": df["message_delivery_time"].mean() + 2 * df["message_delivery_time"].std()
}



# Apply rule-based system
def rule_based_detection(row):
    if (row["packet_drops"] > thresholds["packet_drops"] or
        row["retry_counts"] > thresholds["retry_counts"] or
        row["message_delivery_time"] > thresholds["message_delivery_time"]):
        return 1  # Anomaly
    return 0  # Normal

df['rule_based_flag'] = df.apply(rule_based_detection, axis=1)

# Step 2: Split Data
# Separate data into "flagged by rule-based" and "not flagged"
flagged_data = df[df['rule_based_flag'] == 1].copy()  # Use .copy() to avoid SettingWithCopyWarning
non_flagged_data = df[df['rule_based_flag'] == 0].copy()  # Use .copy() to avoid SettingWithCopyWarning

# Use flagged data for XGBoost training and testing
X = flagged_data[["packet_drops", "retry_counts", "message_delivery_time"]]
y = flagged_data["label"]

# Check if flagged data has both classes
if len(y.unique()) < 2:
    print("Not enough class diversity in flagged data. Using rule-based system for flagged cases.")
    flagged_data['predictions'] = y  # Use the rule-based system's prediction directly
else:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Step 3: Evaluate Hybrid Performance
    print("Hybrid Detection Performance (XGBoost on flagged data):")
    print(classification_report(y_test, y_pred))

    # Add predictions to flagged data
    flagged_data['predictions'] = model.predict(X)

# Non-flagged data is directly considered normal (label = 0)
non_flagged_data['predictions'] = 0

# Combine results
final_results = pd.concat([flagged_data, non_flagged_data])

# Step 4: Evaluate Overall System Performance
print("Overall System Performance:")
print(classification_report(final_results["label"], final_results["predictions"]))
