import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys

from layer import Layer
from network import Network


def main():
    df = pd.read_csv(sys.argv[1])

    y = df.iloc[:, 1].values
    y = np.array([0.00 if label == 'M' else 1.00 for label in y])
    X = df.iloc[:, 2:32].values
    X = np.nan_to_num(X)
    scaler = MinMaxScaler(copy=False)
    X = scaler.fit_transform(X)

    for i in range(10):
        network = Network([
            # Input layer
            Layer(shape=('input_shape', X.shape[1]), activation='relu', input_size=1),
            # Hidden layer 1
            Layer(shape=100, activation='relu', input_size=30),
            # Hidden layer 2
            # Layer(shape=100, activation='relu', input_size=100),
            # # Hidden layer 3
            # Layer(shape=100, activation='relu', input_size=100),
            # Output layer
            Layer(shape=2, activation='relu', input_size=100)],
            # Features and labels
            features=X, labels=y
        )

        # network = Network([
        #     # Input layer
        #     Layer(shape=('input_shape', X.shape[1]), activation='relu', input_size=1),
        #     # Hidden layer 1
        #     Layer(shape=2, activation='relu', input_size=30)],
        #     # Features and labels
        #     features=X, labels=y
        # )

        # print(network.layers[1].nodes[0].activation(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
        # dwdw
        network.fit(epochs=1000, learning_rate=0.01)
        print("Training Finished")
        print("Final Accuracy: ", network.evaluate())
        print("Last Cost: ", network.iterations[-1][1])
        print("Best Cost: ", network.best_cost)
        print("Best Accuracy: ", network.best_prediction)
        print("")
        #for i in range(len(network.layers[1].nodes)):
            #print(sum(network.layers[1].nodes[i].delta))
        """for i in range(len(network.predictions)):
            print(i, " ",  network.predictions[i])"""
        # if network.best_cost < network.iterations[-1][1]:
        network.plot_cost()

    """network2 = Network([
        # Input layer
        Layer(shape=('input_shape', X.shape[1]), activation='relu', input_size=1),
        # Hidden layer 1
        Layer(shape=50, activation='relu', input_size=30),
        # Hidden layer 2
        #Layer(shape=24, activation='relu', input_size=24),
        # # Hidden layer 3
        #Layer(shape=24, activation='relu', input_size=24),
        # Output layer
        Layer(shape=2, activation='softmax', input_size=50)],
        # Features and labels
        features=X, labels=y
    )

    network2.fit(epochs=3000, learning_rate=0.001)
    print("Training Finished")
    print(network2.evaluate())
    print(network2.best_cost)
    print(network2.iterations[-1][1])
    #for i in range(len(network.predictions)):
        print(i, " ",  network.predictions[i])
    network2.plot_cost()

    network3 = Network([
        # Input layer
        Layer(shape=('input_shape', X.shape[1]), activation='relu', input_size=1),
        # Hidden layer 1
        Layer(shape=50, activation='relu', input_size=30),
        # Hidden layer 2
        #Layer(shape=24, activation='relu', input_size=24),
        # # Hidden layer 3
        #Layer(shape=24, activation='relu', input_size=24),
        # Output layer
        Layer(shape=2, activation='softmax', input_size=50)],
        # Features and labels
        features=X, labels=y
    )

    network3.fit(epochs=3000, learning_rate=0.001)
    print("Training Finished")
    print(network3.evaluate())
    print(network3.best_cost)
    print(network3.iterations[-1][1])
    network3.plot_cost()

    network4 = Network([
        # Input layer
        Layer(shape=('input_shape', X.shape[1]), activation='relu', input_size=1),
        # Hidden layer 1
        Layer(shape=50, activation='relu', input_size=30),
        # Hidden layer 2
        #Layer(shape=24, activation='relu', input_size=24),
        # # Hidden layer 3
        #Layer(shape=24, activation='relu', input_size=24),
        # Output layer
        Layer(shape=2, activation='softmax', input_size=50)],
        # Features and labels
        features=X, labels=y
    )

    network4.fit(epochs=3000, learning_rate=0.001)
    print("Training Finished")
    print(network4.evaluate())
    print(network4.best_cost)
    print(network4.iterations[-1][1])
    network4.plot_cost()

    network5 = Network([
        # Input layer
        Layer(shape=('input_shape', X.shape[1]), activation='relu', input_size=1),
        # Hidden layer 1
        Layer(shape=50, activation='relu', input_size=30),
        # Hidden layer 2
        #Layer(shape=24, activation='relu', input_size=24),
        # # Hidden layer 3
        #Layer(shape=24, activation='relu', input_size=24),
        # Output layer
        Layer(shape=2, activation='softmax', input_size=50)],
        # Features and labels
        features=X, labels=y
    )

    network5.fit(epochs=3000, learning_rate=0.001)
    print("Training Finished")
    print(network5.evaluate())
    print(network5.best_cost)
    print(network5.iterations[-1][1])
    network5.plot_cost()"""

    """
    # Define the networks
    networks = [Network([
        # Input layer
        Layer(shape=('input_shape', X.shape[1]), activation='relu', input_size=1),
        # Hidden layer 1
        Layer(shape=20, activation='relu', input_size=30),
        # Hidden layer 2
        Layer(shape=20, activation='relu', input_size=20),
        # Hidden layer 3
        Layer(shape=20, activation='relu', input_size=20),
        # Output layer
        Layer(shape=2, activation='softmax', input_size=20)],
        # Features and labels
        features=X, labels=y
    ) for _ in range(5)]

    for i in range(5):
        networks[i].train_loop(epochs=1200)
        print("Training Finished")
        print(networks[i].evaluate())
    """


if __name__ == '__main__':
    main()
