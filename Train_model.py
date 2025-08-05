import zipfile
import pandas as pd
import os
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import joblib

# Path to the .zip file
zip_path = r"C:\Users\nrebe\Downloads\Data Science\dataSet_IoT_Based_Environmental_Dataset.zip"
extract_path = r"C:\Users\nrebe\Downloads\Data Science\iot_data_extracted"

# Extract the .zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Zip file successfully extracted to:", extract_path)

# List the extracted files
files = os.listdir(extract_path)
print("Files found:", files)

# Try to load the first .csv file found
csv_file = [f for f in files if f.endswith('.csv')][0]
csv_path = os.path.join(extract_path, csv_file)

# Load the dataset
df = pd.read_csv(csv_path)

# Show the first few rows
print("First rows of the dataset:")
print(df.head())

# Show dataset structure and data types
print("\nDataset structure:")
print(df.info())



# Select only the relevant columns for anomaly detection
features = df[['temperature_celsius', 'humidity_percent', 'noise_level_db']].copy()

# Show the first few rows
print("Selected features:")
print(features.head())

#View descriptive statistics
print("\n Descriptive statistics:")
print(features.describe())

#Note: Values above 30 °C may indicate anomalies in an industrial context.

#Note: Humidity values below 30% or above 90% may be considered extreme.

#Note: Noise levels above 75–80 dB are potentially abnormal or hazardous.

# Defining the feature columns to be used
feature_cols = ['temperature_celsius', 'humidity_percent', 'noise_level_db']
X = features[feature_cols]  # Only the original training features

# Create the model
# By setting "Contamination=0.01",the model to assume that 1% of the data are anomalies and to set a threshold based on the anomaly scores to separate the most unusual points.
model = IsolationForest(contamination=0.01, random_state=42)

# Train the model
model.fit(X)

# Get anomaly scores (the more negative, the more likely to be an anomaly)
features['anomaly_score'] = model.decision_function(X)


# Get predictions (1 = normal, -1 = anomaly)
features['anomaly'] = model.predict(X)

# Convert to binary: 1 = anomaly, 0 = normal (for clarity)
features['anomaly'] = features['anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Show some results
print(features[['temperature_celsius', 'humidity_percent', 'noise_level_db', 'anomaly_score', 'anomaly']])

# Counts the number of normal and anomalous data points
# 0 = normal, 1 = anomaly 
anomaly_counts = features['anomaly'].value_counts()

print("Anomaly counts:")
print(anomaly_counts)


# Calculate what percentage of the dataset is classified as anomalies
# We take the mean of the 'anomaly' column (1s = anomalies)
anomaly_percentage = features['anomaly'].mean() * 100

print(f"Anomalies detected: {anomaly_percentage:.2f}% of the dataset")


# Create a scatter plot of temperature vs humidity
# points are colored by whether they were classified as anomalies
plt.figure(figsize=(10, 6))
plt.scatter(
    features['temperature_celsius'],
    features['humidity_percent'],
    c=features['anomaly'],                # Color: 0 = blue, 1 = red (by default in 'coolwarm')
    cmap='coolwarm',
    edgecolor='k'                         # Black edge around points for better visibility
)

plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.title("Anomaly Detection (Red = anomaly, Blue = normal)")
plt.grid(True)
plt.show()


# Save the trained model to a file
joblib.dump(model, 'anomaly_model.pkl')

print("Model saved as 'anomaly_model.pkl'")