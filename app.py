from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

# Load your trained Ridge model
with open('model_ridge.pkl', 'rb') as model_file:
    model_ridge = pickle.load(model_file)

# Load the scaler used during training
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

# Serve the HTML file
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Extract and log-transform the 'Absences' feature
        log_absences = np.log1p(data['Absences'])

        # Extract the features in the right order, including log-transformed 'Absences'
        features = np.array([
            data['ParentalEducation'], 
            data['StudyTimeWeekly'], 
            data['Tutoring'], 
            data['ParentalSupport'], 
            data['Extracurricular'], 
            data['Sports'], 
            data['Music'], 
            data['Volunteering'],
            log_absences  # Using log-transformed 'Absences'
        ]).reshape(1, -1)

        # Scale the features using the loaded scaler
        features_scaled = scaler.transform(features)

        # Predict using the Ridge Regression model
        prediction_ridge = model_ridge.predict(features_scaled)
        prediction_ridge = np.clip(prediction_ridge, 2.0, 4.0)  # Ensure GPA is within the valid range

        # Send back the prediction from the Ridge model
        return jsonify({
            'prediction_ridge': prediction_ridge.tolist(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
