from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model/model.pkl")
cols = joblib.load("model/columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    present_price = float(request.form["present_price"])
    kms_driven = float(request.form["kms_driven"])
    owner = int(request.form["owner"])
    car_age = float(request.form["car_age"])
    fuel = request.form["fuel"]
    seller = request.form["seller"]
    transmission = request.form["transmission"]

    data = pd.DataFrame(
        [[present_price, kms_driven, owner, car_age]],
        columns=["Present_Price", "Kms_Driven", "Owner", "Car_Age"]
    )

    for c in cols:
        if c not in data.columns:
            data[c] = 0

    if fuel == "Diesel":
        data["Fuel_Type_Diesel"] = 1

    if fuel == "Petrol":
        data["Fuel_Type_Petrol"] = 1

    if seller == "Individual":
        data["Seller_Type_Individual"] = 1

    if transmission == "Manual":
        data["Transmission_Manual"] = 1

    data = data[cols]

    prediction = model.predict(data)[0]

    return render_template(
        "index.html",
        prediction_text=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)