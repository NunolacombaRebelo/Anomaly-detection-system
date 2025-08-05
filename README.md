# Anomaly Detection System for IoT Sensor Data

This project implements a real-time anomaly detection system for IoT sensor data in an industrial environment. It uses sensor readings such as temperature, humidity, and noise to detect abnormal behavior using a simple machine learning model (Isolation Forest).

---

##  Project Overview

A modern factory equipped with multiple IoT sensors collects continuous data. This system is designed to:

- Simulate live sensor readings (temperature, humidity, noise)
- Detect anomalies in real-time using a trained model
- Expose predictions through a REST API (Flask)
- Log results for further monitoring

---

##  Machine Learning Model

- **Algorithm:** Isolation Forest  
- **Features used:** 
  - `temperature_celsius`
  - `humidity_percent`
  - `noise_level_db`
- **Contamination rate:** 1% (i.e., 1% of data is assumed to be anomalies)
- **Model Output:**
  - `anomaly_score`: numeric anomaly score (lower = more abnormal)
  - `is_anomaly`: binary classification (1 = anomaly, 0 = normal)

---

##  Project Structure

- `app.py` — Flask API serving the trained model  
- `train_model.py` — Script to train and evaluate the ML model  
- `sensor_simulator.py` — Simulates live sensor data and sends it to the API  
- `anomaly_model.pkl` — Saved trained model *(not uploaded to GitHub)*  
- `README.md` — Project documentation  
- `venv/` — Virtual environment *(not uploaded to GitHub)*


---

## How to Run the Project

### 1. Install dependencies

Activate your virtual environment and run:

```bash
pip install pandas scikit-learn matplotlib joblib flask requests

```

## 2. Train the Model, Start the Flask API, and Simulate Sensor Data

Follow these steps in separate terminals:

1. Train and save the model:

```bash
python train_model.py
```

2. Start the Flask API:

```bash
python app.py
```
The API will be available at: [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)

3. In another terminal, simulate the sensor data:

```bash
python sensor_simulator.py
```

##  Example API Response

**Request:**

```json
{
  "temperature": 25.0,
  "humidity": 60.0,
  "noise": 55.0
}
```

**Returns:**

```json
{
  "anomaly_score": 0.2197,
  "is_anomaly": 0
}
```

---

##  Model Details

- **Algorithm:** Isolation Forest  
- **Features used:** `temperature`, `humidity`, `noise`  
- **Contamination:** 1% (assumes 1% of data is anomalous)

---

##  License

This project is for academic purposes only

  

