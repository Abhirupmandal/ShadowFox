import pandas as pd


def predict(model):

    data = {
        "Sales": [0],
        "Quantity": [2],
        "Discount": [0.2],
        "Profit": [50],
        "Category": ["Furniture"],
        "Region": ["East"],
        "Segment": ["Consumer"],
    }

    df = pd.DataFrame(data)

    df = pd.get_dummies(
        df,
        columns=[
            "Category",
            "Region",
            "Segment"
        ]
    )

    df = df.drop("Sales", axis=1)

    df = df.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    p = model.predict(df)

    print("Prediction:", p)