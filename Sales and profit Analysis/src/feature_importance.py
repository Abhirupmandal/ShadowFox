import matplotlib.pyplot as plt

def importance(model, X):

    imp = model.feature_importances_

    names = X.columns

    plt.figure(figsize=(8,5))

    plt.barh(names, imp)

    plt.title("Feature Importance")

    plt.savefig("logs/importance.png")

    plt.show()