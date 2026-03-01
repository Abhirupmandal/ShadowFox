from sklearn.ensemble import RandomForestRegressor

def get_model():

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )

    return model