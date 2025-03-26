import pandas as pd
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("iot_telemetry_data.csv")

# Calculate thresholds
thresholds = {
    "packet_drops": df["packet_drops"].mean() + 3 * df["packet_drops"].std(),
    "retry_counts": df["retry_counts"].mean() + 2.5 * df["retry_counts"].std(),
    "message_delivery_time": df["message_delivery_time"].mean() + 3 * df["message_delivery_time"].std()
}

# Rule-based detection
def rule_based_detection(row):
    if (row["packet_drops"] > thresholds["packet_drops"] or
        row["retry_counts"] > thresholds["retry_counts"] or
        row["message_delivery_time"] > thresholds["message_delivery_time"]):
        return 1  # Anomaly
    return 0  # Normal

df['rule_based_flag'] = df.apply(rule_based_detection, axis=1)

# Evaluate performance
print("Rule-Based Detection Performance:")
print(classification_report(df['label'], df['rule_based_flag']))
