from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model/model.pkl")
cols = joblib.load("model/columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        present_price = float(request.form.get("present_price") or 0)
        kms_driven = float(request.form.get("kms_driven") or 0)
        owner = int(request.form.get("owner") or 0)
        car_age = float(request.form.get("car_age") or 0)

        fuel = request.form.get("fuel")
        seller = request.form.get("seller")
        transmission = request.form.get("transmission")

    except ValueError:
        return render_template(
            "index.html",
            prediction_text="Invalid input. Please enter valid numbers."
        )

    data = pd.DataFrame(
        [[present_price, kms_driven, owner, car_age]],
        columns=["Present_Price", "Kms_Driven", "Owner", "Car_Age"]
    )

    for col in cols:
        if col not in data.columns:
            data[col] = 0

    if "Fuel_Type_Diesel" in cols and fuel == "Diesel":
        data["Fuel_Type_Diesel"] = 1

    if "Fuel_Type_Petrol" in cols and fuel == "Petrol":
        data["Fuel_Type_Petrol"] = 1

    if "Seller_Type_Individual" in cols and seller == "Individual":
        data["Seller_Type_Individual"] = 1

    if "Transmission_Manual" in cols and transmission == "Manual":
        data["Transmission_Manual"] = 1

    data = data[cols]

    prediction = model.predict(data)[0]
    prediction_value = round(prediction, 2)

    return render_template(
        "index.html",
        prediction_text=prediction_value
    )


if __name__ == "__main__":
    app.run(debug=True)