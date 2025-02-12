import joblib
import numpy as np

# Load models
try:
    classification_model = joblib.load('classification_model.joblib')
    redshift_model = joblib.load('redshift_model.joblib')
    anomaly_model = joblib.load('anomaly_model.joblib')
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

def get_prediction(case, input_data):
    try:
        if case == '1':  # Case 1: Object Classification
            required_features = ['u', 'g', 'r', 'i', 'z']
            input_data = [float(input_data.get(feat, 0)) for feat in required_features]
            prediction = classification_model.predict([input_data])
            return f"Predicted Class: {int(prediction[0])}"

        elif case == '2':  # Case 2: Photometric Redshift Binning
            required_features = ['u', 'g', 'r', 'i', 'z']
            input_data = [float(input_data.get(feat, 0)) for feat in required_features]
            prediction = redshift_model.predict([input_data])
            bins = {0: 'Low Redshift', 1: 'Medium Redshift', 2: 'High Redshift'}
            return f"Predicted Redshift Bin: {bins.get(int(prediction[0]), 'Unknown')}"

        elif case == '3':  # Case 3: Satellite Anomaly Detection
            required_features = ['alpha', 'delta', 'fiber_ID']
            input_data = [float(input_data.get(feat, 0)) for feat in required_features]
            prediction = anomaly_model.predict([input_data])
            anomaly_status = 'Anomalous' if prediction[0] == 1 else 'Normal'
            return f"Satellite Observation: {anomaly_status}"

        else:
            return "Invalid case selected."
    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == '__main__':
    print("Welcome to the Prediction Tester!")
    print("Select a case:")
    print("1: Object Classification")
    print("2: Photometric Redshift Binning")
    print("3: Satellite Anomaly Detection")
    case = input("Enter the case number (1/2/3): ")

    # Get user inputs based on the selected case
    if case == '1' or case == '2':
        print("Enter the following input features:")
        input_data = {
            'u': input("u: "),
            'g': input("g: "),
            'r': input("r: "),
            'i': input("i: "),
            'z': input("z: "),
        }
    elif case == '3':
        print("Enter the following input features:")
        input_data = {
            'alpha': input("alpha: "),
            'delta': input("delta: "),
            'fiber_ID': input("fiber_ID: "),
        }
    else:
        print("Invalid case selected. Exiting.")
        exit()

    # Get and print prediction
    result = get_prediction(case, input_data)
    print("\nPrediction Result:")
    print(result)
