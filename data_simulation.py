import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(62)

# Generate normal traffic data
n_normal = 900
normal_data = {
    "packet_drops": np.random.poisson(2, n_normal),
    "retry_counts": np.random.poisson(3, n_normal),
    "message_delivery_time": np.random.normal(50, 5, n_normal)
}

# Generate attack traffic data
n_attack = 200
attack_data = {
    "packet_drops": np.random.poisson(10, n_attack),
    "retry_counts": np.random.poisson(15, n_attack),
    "message_delivery_time": np.random.normal(100, 10, n_attack)
}

# Combine data and save to CSV
normal_df = pd.DataFrame(normal_data)
attack_df = pd.DataFrame(attack_data)
normal_df['label'] = 0
attack_df['label'] = 1
dataset = pd.concat([normal_df, attack_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
dataset.to_csv("iot_telemetry_data.csv", index=False)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.hist(dataset[dataset['label'] == 0]['message_delivery_time'], bins=20, alpha=0.7, label="Normal")
plt.hist(dataset[dataset['label'] == 1]['message_delivery_time'], bins=20, alpha=0.7, label="Attack")
plt.title("Message Delivery Time Distribution")
plt.xlabel("Message Delivery Time (ms)")
plt.ylabel("Frequency")
plt.legend()
plt.show()
