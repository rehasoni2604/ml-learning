#This script is a Flask web application used for serving machine learning models. 
# It loads pre-trained models, processes input from a web interface, and returns predictions. 
# The models perform different tasks based on the selected case: object classification, photometric redshift binning, and satellite anomaly detection.

#Here’s a breakdown of what each section of the code does:




#1. Importing Necessary Libraries

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np


#2. Initializing the Flask Application
app = Flask(__name__)


#3. Loading Pre-trained Models (Pipelines)
try:
    classification_pipeline = joblib.load('classification_model.joblib')
    redshift_pipeline = joblib.load('redshift_model.joblib')
    anomaly_pipeline = joblib.load('anomaly_model.joblib')
except Exception as e:
    print(f"Error loading pipelines: {e}")



#4. Defining Routes and View Functions
@app.route('/')
def index():
    return render_template('index.html')



#4.2 Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    case = request.form['case']
    features = request.form.to_dict()


#5. Making Predictions Based on the Selected Case    
    try:
        #5.1 Case 1: Object Classification
        if case == '1':  # Object Classification
            required_features = ['u', 'g', 'r', 'i', 'z']
            input_data = np.array([[float(features.get(feat, 0)) for feat in required_features]])
            prediction = classification_pipeline.predict(input_data)
            result = f"Predicted Class: {int(prediction[0])}"

        #5.2 Case 2: Photometric Redshift Binning
        elif case == '2':  # Photometric Redshift Binning
            required_features = ['u', 'g', 'r', 'i', 'z']
            input_data = np.array([[float(features.get(feat, 0)) for feat in required_features]])
            prediction = redshift_pipeline.predict(input_data)
            bins = {0: 'Low Redshift', 1: 'Medium Redshift', 2: 'High Redshift'}
            result = f"Predicted Redshift Bin: {bins.get(int(prediction[0]), 'Unknown')}"

        #5.3 Case 3: Satellite Anomaly Detection
        elif case == '3':  # Satellite Anomaly Detection
            required_features = ['alpha', 'delta', 'fiber_ID']
            input_data = np.array([[float(features.get(feat, 0)) for feat in required_features]])
            prediction = anomaly_pipeline.predict(input_data)
            anomaly_status = 'Anomalous' if prediction[0] == 1 else 'Normal'
            result = f"Satellite Observation: {anomaly_status}"

#6. Handling Invalid Case or Errors
        else:
            result = "Invalid case selected."

        return result
    
#Error Handling
    except Exception as e:
        return f"Error: {str(e)}"
    
#7. Running the Flask App
if __name__ == '__main__':
    app.run(debug=True, port=5001)






    #Summary
#This Flask application serves pre-trained machine learning models for three different tasks:

#~ Object Classification (Predicting the class of astronomical objects based on u, g, r, i, z values).
#~ Photometric Redshift Binning (Classifying redshift values into categories).
#~ Satellite Anomaly Detection (Detecting anomalies based on alpha, delta, and fiber_ID).
