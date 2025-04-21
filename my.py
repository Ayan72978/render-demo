from flask import Flask, request, render_template
import pickle
import numpy as np
import os
# Load the trained models from the pickle file
with open(r"rainfall_models.pkl", "rb") as model_file:
    models = pickle.load(model_file)

# Select one model (e.g., the first model in the list)
model = models[0]  # Example: LogisticRegression

# If you have a scaler, you can load it like this:
# with open('scaler.pkl', 'rb') as scaler_file:
#     scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [
            float(request.form['MinTemp']),
            float(request.form['MaxTemp']),
            float(request.form['Rainfall']),
            float(request.form['Evaporation']),
            float(request.form['Sunshine']),
            float(request.form['WindGustSpeed']),
            float(request.form['WindSpeed9am']),
            float(request.form['WindSpeed3pm']),
            float(request.form['Humidity9am']),
            float(request.form['Humidity3pm']),
            float(request.form['Pressure9am']),
            float(request.form['Pressure3pm']),
            float(request.form['Temp9am']),
            float(request.form['Temp3pm']),
            float(request.form['Cloud9am']),
            float(request.form['Cloud3pm'])
        ]

        input_array = np.array([input_features])

        # Uncomment if you're using a scaler
        # input_array = scaler.transform(input_array)

        prediction = model.predict(input_array)
        output = 'Rainfall Expected' if prediction[0] == 1 else 'No Rainfall Expected'
        return render_template('index.html', prediction_text=f'Prediction: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
