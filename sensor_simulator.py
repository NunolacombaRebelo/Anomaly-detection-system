import requests
import random
import time

# URL of the Flask API
API_URL = "http://127.0.0.1:5000/predict"

def simulate_sensor_data():
    # Generate  random values
    temperature = round(random.uniform(15.0, 35.0), 2)
    humidity = round(random.uniform(20.0, 95.0), 2)
    noise = round(random.uniform(40.0, 90.0), 2)
    return {
        "temperature": temperature,
        "humidity": humidity,
        "noise": noise
    }

print("Simulating sensor data... (Press Ctrl+C to stop)")

while True:
    # Generate new sensor reading
    sensor_data = simulate_sensor_data()
    
    # Send the data to the API
    try:
        response = requests.post(API_URL, json=sensor_data)
        result = response.json()

        # Print sent data and prediction response
        print(f"\n Sent: {sensor_data}")
        print(f"Response: {result}")

    except Exception as e:
        print(f"Error sending data: {e}")

    # Wait time before sending the next reading
    time.sleep(2)
