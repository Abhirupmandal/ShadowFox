from src.model import get_model

def train(X_train, y_train):
    model = get_model()
    model.fit(X_train, y_train)
    return model