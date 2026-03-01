import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("dataset/car.csv")

df["Car_Age"] = 2024 - df["Year"]

df.drop(["Car_Name", "Year"], axis=1, inplace=True)

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

model = RandomForestRegressor(n_estimators=200)

model.fit(X, y)

joblib.dump(model, "model/model.pkl")

joblib.dump(list(X.columns), "model/columns.pkl")

print("done")