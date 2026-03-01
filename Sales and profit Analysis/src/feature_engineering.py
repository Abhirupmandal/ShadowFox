import pandas as pd

def feature_engineering(df):

    df = df.copy()

    df = df[
        [
            "Sales",
            "Quantity",
            "Discount",
            "Profit",
            "Category",
            "Region",
            "Segment"
        ]
    ]

    df = pd.get_dummies(
        df,
        columns=[
            "Category",
            "Region",
            "Segment"
        ]
    )

    df = df.dropna()

    return df