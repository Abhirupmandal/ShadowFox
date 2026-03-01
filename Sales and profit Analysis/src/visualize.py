import matplotlib.pyplot as plt

def plot(df):

    plt.figure(figsize=(10,5))

    plt.plot(df["Sales"])

    plt.title("Sales Data")

    plt.savefig("logs/sales_graph.png")

    plt.show()