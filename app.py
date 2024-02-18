import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create a Flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))

# Define a function to preprocess input data
def preprocess_input(data):
    # Instantiate a LabelEncoder for encoding categorical columns
    label_encoder = LabelEncoder()

    # Convert categorical columns (Geography, Gender) into numerical
    data['Geography'] = label_encoder.fit_transform(data['Geography'])
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    return data

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the input values from the form and store them in a dictionary
        input_data = {
            'CreditScore': float(request.form['CreditScore']),
            'Geography': request.form['Geography'],
            'Gender': request.form['Gender'],
            'Age': float(request.form['Age']),
            'Tenure': float(request.form['Tenure']),
            'Balance': float(request.form['Balance']),
            'NumOfProducts': float(request.form['NumOfProducts']),
            'HasCrCard': float(request.form['HasCrCard']),
            'IsActiveMember': float(request.form['IsActiveMember']),
            'EstimatedSalary': float(request.form['EstimatedSalary'])
        }

        # Convert the dictionary to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data
        input_df = preprocess_input(input_df)

        # Make a prediction using the model
        prediction = model.predict(input_df)

        # Display the prediction
        return render_template("index.html", prediction_text="The prediction is {}".format(prediction[0][0]))

    except Exception as e:
        return render_template("index.html", prediction_text="Error: {}".format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)
