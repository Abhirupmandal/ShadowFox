from src.loader import load_data
from src.feature_engineering import feature_engineering
from src.preprocessing import split_data
from src.train import train
from src.evaluate import evaluate
from src.save_model import save, load
from src.predict import predict
from src.visualize import plot
from src.utils import log
from src.feature_importance import importance

log("start")

df = load_data()

df = feature_engineering(df)

plot(df)

X_train, X_test, y_train, y_test = split_data(df)

model = train(X_train, y_train)

evaluate(model, X_test, y_test)

importance(model, X_train)

save(model)

m = load()

predict(m)

log("done")