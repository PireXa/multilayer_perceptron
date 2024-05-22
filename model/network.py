import numpy as np
import matplotlib.pyplot as plt


class Network:
    """
    multilayer-perceptron model
    """

    def __init__(self, layers, features, labels):
        self.layers = layers
        self.features = features
        self.labels = labels
        self.predictions = None
        self.learning_rate = 0.001
        self.iterations = []
        self.gradients = []
        self.predictions = []
        self.best_cost = 1.00
        self.best_prediction = 0.00
        self.last_prediction = 0.00
        self.last_cost = 1.00
        self.label_list = []
        self.loss = []

    def print_network(self):
        for i in range(len(self.layers)):
            if i == 1:
                # print(i)
                print(self.layers[i].nodes[0].weights[23])

    def feed_input(self):
        """
        Gets input features to the network's input layer
        """
        # print(self.features.shape)
        for i in range(len(self.layers[0].nodes)):
            self.layers[0].nodes[i].output = self.features[:, i]
        # print('dedede', len(self.features))
        return self.features

    def compute_cost(self):
            """
            Function to compute the cost of the network using the following formula:
            (y - ŷ)^2 / 2
            """
            n = len(self.labels)
            # print(n)
            # print(self.predictions)

            # cost = np.sum((self.labels - self.predictions) ** 2) / n
            cost = np.sum((self.predictions - self.labels) ** 2) / n
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_prediction = self.evaluate()
            return cost

    def compute_cost_cross_entropy(self):
        """
        Function to compute the cost of the network using the following formula:
        -y log(ŷ) - (1 - y) log(1 - ŷ)
        """
        n = len(self.labels)

        # print('output', self.layers[-1].nodes[1].output)
        # print('predictions', self.predictions)
        # print('labels', self.labels)
        # cost = -1/n * np.sum(self.labels * np.log(self.predictions + 1e15) + (1 - self.labels) * np.log(1 - self.predictions + 1e15))
        cost = np.mean(-self.labels * np.log(self.predictions + 1e-15) - (1 - self.labels) * np.log(1 - self.predictions + 1e-15))
        # print('cost', cost)
        return cost

    def feedforward(self):
        input = self.feed_input()
        # print('Input')
        # print('input', len(input))
        for i in range(1, len(self.layers) - 1):
            # print('Layer', i)
            next_input = []
            for j in range(len(self.layers[i].nodes)):
                # weighted_sum = np.dot(input, self.layers[i].nodes[j].weights.T)
                # print('input', input.T)
                weighted_sum = np.dot(self.layers[i].nodes[j].weights, input.T)
                # print('Weighted Sum', weighted_sum)
                self.layers[i].nodes[j].output = self.layers[i].nodes[j].activation(weighted_sum)
                # print('dede1', i, j, len(self.layers[i].nodes[j].output))
                # for k in range(len(input)):
                #     print('input', input[k])
                #     print('weights', self.layers[i].nodes[j].weights)
                    # weighted_sum = np.dot(input[k], self.layers[i].nodes[j].weights)
                    # print('Weighted Sum', weighted_sum)
                #     print('Activation', self.layers[i].nodes[j].activation(weighted_sum))
                    # self.layers[i].nodes[j].output.append(self.layers[i].nodes[j].activation(weighted_sum))
                    # print('Output', i, j, k)
                    # print(self.layers[i].nodes[j].output)
                    # print(sum(self.layers[i].nodes[j].output))
                    # print(i, j, k)
                    # if k == 1 and j == 1 and i == 2:
                        # brdedddee
                # if j == 1 and i == 2:
                    # print('Output')
                    # print(self.layers[2].nodes[0].output)
                    # print(self.layers[2].nodes[1].output)
                    # dwdedewde
                # print('dede2', i, j, len(self.layers[i].nodes[j].output))
                next_input.append(self.layers[i].nodes[j].output)
            # print('Next Input', len(next_input))
            # print('Next Input', len(next_input[0]))
            input = np.array(next_input).T
            # print('Input', input.shape)
            # print('Output')
            # print(sum(input))
            # print(input)
        # dwe
        #transform the output layer into predictions of 1 or 0
        # print('Output')
        # print(input[5])
        # output_weighted_sum = [np.dot(self.layers[-1].nodes[0].weights, input.T), np.dot(self.layers[-1].nodes[1].weights, input.T)]
        # print('Output Weighted Sum')
        # print(output_weighted_sum)
        # for i in range(len(output_weighted_sum[0])):

        #run softmax on the output layer
        # sum = 0
        # print('input', input)
        # print('input', input[0])
        weighted_sum_1 = np.dot(self.layers[-1].nodes[0].weights, input.T)
        weighted_sum_2 = np.dot(self.layers[-1].nodes[1].weights, input.T)
        # print('weighted_sum_1', weighted_sum_1)
        # print('weighted_sum_2', weighted_sum_2)
        output_node1 = weighted_sum_1
        output_node2 = weighted_sum_2
        # print('output_node1', output_node1)
        # print('output_node2', output_node2)

        pairs = np.array(list(zip(output_node1, output_node2)))
        # print('pairs', pairs)

        # output_node1 = output_node1 - np.max(output_node1)
        # output_node1 = np.exp(output_node1)
        # output_node1 = output_node1 / np.sum(output_node1)
        # self.layers[-1].nodes[0].output = output_node1
        # self.predictions = np.array([0 if output_node1[i] > 0.5 else 1 for i in range(len(output_node1))])

        for i in range(len(pairs)):
            # print('pair', pairs[i])
            # print('max', np.max(pairs[i]))
            pairs[i] = pairs[i] - np.max(pairs[i])
            # print('pair', pairs[i])
            pairs[i] = np.exp(pairs[i])
            # print('exp pair', pairs[i])
            pairs[i] = pairs[i] / np.sum(pairs[i])
            # print('softmax pair', pairs[i])

        # print('softmaxed pairs', pairs)
        softmax_values_1 = pairs[:, 0]
        softmax_values_2 = pairs[:, 1]
        # print('softmax_values_1', softmax_values_1)
        # print('softmax_values_2', softmax_values_2)
        self.layers[-1].nodes[0].output = softmax_values_1
        self.layers[-1].nodes[1].output = softmax_values_2
        # print('softmax_values_1', softmax_values_1)
        # print('softmax_values_2', softmax_values_2)

        self.predictions = []
        # print(len(self.predictions))
        # print('len input', input.shape)
        # self.predictions = np.array([1 if output[0] > output[1] else 0 for output in input])
        self.predictions = np.array([0 if softmax_values_1[i] > softmax_values_2[i] else 1 for i in range(len(softmax_values_1))])
        # print('Predictions: ', self.predictions)
        # print(input)

    def backpropagation(self):
        # print('Predictions', self.predictions)
        for i in range(len(self.layers[-1].nodes)):
            self.label_list = np.array(self.label_list)
            prediction = np.clip(self.layers[-1].nodes[i].output, 1e-15, 1 - 1e-15)
            # prediction = self.layers[-1].nodes[i].output
            # print('prediction', prediction)
            # print('label', self.label_list[:, i])
            # print('single predictions', prediction * self.label_list[:, i])
            # print('dot', prediction @ self.label_list[:, i])
            # loss = -(np.log(prediction) @ self.label_list[:, i])
            # print('loss', loss)
            # loss -= 1
            # loss /= len(self.label_list)
            # print('average loss', np.mean(loss))
            # print('loss 2', loss / len(self.label_list))
            # print('derivative', self.layers[-1].nodes[i].activation_derivative(self.layers[-1].nodes[i].output))

            # print('label', self.label_list[:, i])
            # print('prediction', prediction)
            # make an array with all the predictions where the label is 1
            # true_predictions = prediction[self.label_list[:, i] == 1]
            # true_labels = self.label_list[:, i][self.label_list[:, i] == 1]
            # print('true predictions', true_predictions)
            # print('true labels', true_labels)

            self.layers[-1].nodes[i].delta = prediction - self.label_list[:, i]
            # self.layers[-1].nodes[i].delta = - (self.label_list[:, i] / prediction) + (1 - self.label_list[:, i]) / (1 - prediction)
            # self.layers[-1].nodes[i].delta = true_predictions - true_labels
            # self.layers[-1].nodes[i].delta = -np.log(prediction) @ self.label_list[:, i]
            # self.layers[-1].nodes[i].delta = self.label_list[:, i] - self.label_list[:, i]
            # print('delta', self.layers[-1].nodes[i].delta)
            # self.layers[-1].nodes[i].delta = np.mean(self.layers[-1].nodes[i].delta)
            # print('delta', self.layers[-1].nodes[i].delta)
            # print('\n\n')

        for i in range(len(self.layers) - 2, 0, -1):
            for j in range(len(self.layers[i].nodes)):
                activation_derivative = self.layers[i].nodes[j].activation_derivative(self.layers[i].nodes[j].output)
                # print(self.layers[i + 1].nodes[j].weights.shape, self.layers[i + 1].nodes[j].delta.shape)
                gradient = 0
                for k in range(len(self.layers[i + 1].nodes)):
                    gradient += self.layers[i + 1].nodes[k].weights[j] * self.layers[i + 1].nodes[k].delta * activation_derivative
                # self.layers[i].nodes[j].delta = activation_derivative * sum
                # print('gradient', gradient)
                # dd
                self.layers[i].nodes[j].delta = gradient

    def update_weights(self):
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].nodes)):
                for k in range(len(self.layers[i].nodes[j].weights)):
                    # print(i, j, k)
                    # print(self.learning_rate, self.layers[i].nodes[j].delta)
                    # print(self.layers[i - 1].nodes[j].delta.shape)
                    # print(self.layers[i].nodes[j].delta.shape, self.layers[i - 1].nodes[j].output.shape)
                    step = np.dot(self.layers[i].nodes[j].delta, self.layers[i - 1].nodes[k].output)
                    # print('step', step)
                    # print('delta', self.layers[i].nodes[j].delta)
                    # print('output', self.layers[i - 1].nodes[k].output)
                    # step = self.layers[i].nodes[j].delta @ self.layers[i - 1].nodes[k].output
                    # print('step', step)
                    # step = step / len(self.layers[i].nodes[j].delta)
                    # print('step', step)

                    # print('step', step)
                    # print(self.layers[i].nodes[j].weights[k])
                    # print('learning rate', self.learning_rate)
                    # print('delta', self.layers[i].nodes[j].delta)
                    # if i == 2 and j == 0:
                    # print('weights', k, self.layers[i].nodes[j].weights[k])
                    # print('step', step)
                    # print('learning rate', self.learning_rate)
                    self.layers[i].nodes[j].weights[k] -= self.learning_rate * step
                    # print('weights', k, self.layers[i].nodes[j].weights[k])
                    # self.layers[i].nodes[j].weights[k] -= self.learning_rate * np.mean(self.layers[i].nodes[j].delta)
                    # if i == 2 and j == 0:
                    #     print('weights', k, self.layers[i].nodes[j].weights[k])
        

    def fit(self, epochs: int, learning_rate: float):
        for i in range(len(self.labels)):
            if self.labels[i] == 0:
                self.label_list.append([1, 0])
            else:
                self.label_list.append([0, 1])
        # print('Labels', self.label_list)
        print("Training Started")
        print('Examples', len(self.features))
        for i in range(epochs):
            # print('Epoch', i)
            # for j in range(len(self.layers)):
                # print('Layer', j)
                # print('Nodes', len(self.layers[j].nodes))
            self.feedforward()
            if i % 10 == 0:
                print("Epoch: ", i, " Cost: ", self.compute_cost())
                print("Cross Entropy Cost: ", self.compute_cost_cross_entropy())
                # print("\n\n")
            # d3
            self.iterations.append((i, self.compute_cost()))
            self.backpropagation()
            self.update_weights()
            self.loss.append((i, np.abs(np.mean(self.layers[-1].nodes[0].delta))))
            # print('predicts', self.predictions)
            # print('\n\n\n')
            # if i == 1000:
            #     self.plot_error()
            #     dd
            # for j in range(len(self.layers)):
                # for k in range(len(self.layers[j].nodes)):
                    # self.layers[j].nodes[k].output = []
        # print('self.predictions', self.predictions)

    def evaluate(self):
        """
        Evaluate the accuracy of the model
        """
        # Compare the predicted classes with the true classes
        accuracy = np.mean(self.predictions == self.labels)
        return accuracy
    
    def plot_cost(self):
        """
        Plot the cost evaluation
        """
        x = [item[0] for item in self.iterations]
        y = [item[1] for item in self.iterations]

        # Plot the data as a scatter graph
        plt.scatter(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Cost evaluation')
        plt.grid(True)
        plt.show()

    def plot_loss(self):
        """
        Plot the cost evaluation
        """
        x = [item[0] for item in self.loss]
        y = [item[1] for item in self.loss]

        # Plot the data as a scatter graph
        plt.scatter(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Error evaluation')
        plt.grid(True)
        plt.show()

    def __str__(self):
        # Print each one of the layers of the network
        return '\n'.join([str(layer) for layer in self.layers])
