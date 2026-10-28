flask_app_code = """
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and scaler
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Expecting inputs as JSON: {"Cycle": value, "Voltage": value, "Temperature": value}
        cycle = data['Cycle']
        voltage = data['Voltage']
        temperature = data['Temperature']

        # Format features for model input
        features = np.array([[cycle, voltage, temperature]])

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])

        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'prediction_label': 'End-of-Life (EOL)' if prediction == 1 else 'Normal',
            'eol_probability': probability
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""

# Write the flask app code to an app.py file
with open('app.py', 'w') as f:
    f.write(flask_app_code)

print("Flask API script has been successfully created and written to 'app.py'!")

