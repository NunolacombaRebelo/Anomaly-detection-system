from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('anomaly_model.pkl')

# Create Flask app
app = Flask(__name__)

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON data
        data = request.get_json()

        # Extract features from JSON
        temperature = data['temperature']
        humidity = data['humidity']
        noise = data['noise']

        # Format data for prediction
        input_data = np.array([[temperature, humidity, noise]])

        # Get score and prediction
        score = model.decision_function(input_data)[0]
        prediction = int(model.predict(input_data)[0] == -1)  # 1 = anomaly, 0 = normal

        return jsonify({
            'anomaly_score': score,
            'is_anomaly': prediction
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
