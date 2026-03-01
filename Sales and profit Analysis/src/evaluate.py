from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def evaluate(model, X_test, y_test):

    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)

    mse = mean_squared_error(y_test, pred)

    rmse = np.sqrt(mse)

    print("R2:", r2)
    print("MSE:", mse)
    print("RMSE:", rmse)