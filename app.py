from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

# Load saved models and encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Predefined categories for dropdown menus
brand_options = label_encoders['brand'].classes_.tolist()
car_model_options = label_encoders['car_model'].classes_.tolist()
transmission_options = label_encoders['transmission'].classes_.tolist()
body_type_options = label_encoders['body_type'].classes_.tolist()
fuel_type_options = label_encoders['fuel_type'].classes_.tolist()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            # Extract form data
            input_data = {
                "brand": request.form["brand"],
                "car_model": request.form["car_model"],
                "model_year": float(request.form["model_year"]),  # Fix name to match training
                "transmission": request.form["transmission"],
                "body_type": request.form["body_type"],
                "fuel_type": request.form["fuel_type"],
                "engine_capacity": float(request.form["engine_capacity"]),
                "kilometers_run": float(request.form["kilometers_run"])
            }

            # Encode categorical values
            for col in label_encoders:
                if col in input_data:
                    input_data[col] = label_encoders[col].transform([input_data[col]])[0]

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Scale the input features
            input_scaled = scaler.transform(input_df)

            # Make prediction
            predicted_price = model.predict(input_scaled)[0]
            prediction = round(predicted_price, 2)
        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error,
                           brand_options=brand_options,
                           car_model_options=car_model_options,
                           transmission_options=transmission_options,
                           body_type_options=body_type_options,
                           fuel_type_options=fuel_type_options)

if __name__ == '__main__':
    app.run(debug=True)