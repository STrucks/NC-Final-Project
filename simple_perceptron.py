import random as ran
import numpy as np
import math


def load_iris():
    data = [line.replace("\n", "").split(",") for line in open("iris_data.txt", 'r').readlines()]
    data1 = [[float(line[0]), float(line[1]), float(line[2]), float(line[3]), "0"] for line in data if
            line[4] == "Iris-setosa"]
    data1 += [[float(line[0]), float(line[1]), float(line[2]), float(line[3]), "1"] for line in data if
            line[4] == "Iris-versicolor"]
    return data1


def load_OR():
    data = [line.replace("\n", "").split(",") for line in open("OR.txt", 'r').readlines()]
    data = [[float(line[0]), float(line[1]),line[2]] for line in data]

    return data


def load_XOR():
    data = [line.replace("\n", "").split(",") for line in open("XOR.txt", 'r').readlines()]
    data = [[float(line[0]), float(line[1]), line[2]] for line in data]
    return data

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


if __name__ == '__main__':
    N = 4
    learning_rate = 0.01
    # we define the perceptron to have 4 input nodes and random weights:
    input_nodes = np.asarray([1] + [0]*N)
    weights = np.random.rand(N+1,1) - 0.5
    # since simple perceptron can only distinguish 2 classes, we only take the first 100 data points
    # data = load_iris()[0:100]
    data = load_iris()
    np.random.shuffle(data)
    # network output with sigmoid(np.matmul(input_nodes, weights))
    # print(sigmoid(np.matmul(input_nodes, weights)))

    # start learning phase
    for iter in range(0,100):
        nr_incorrect = 0
        for row in data:
            # predict the outcome of the network for the row:
            input_nodes = np.asarray([1] + row[0:N])
            update = [0] * (N + 1)
            # for each node:
            y = sigmoid(np.matmul(input_nodes, weights))
            print(y)
            if y > 0.5 and row[N] == "0":
                nr_incorrect += 1
                for node in range(N):
                    # tweak weights:
                    update[node] = learning_rate * (0 - 1) * input_nodes[node]
            if y <= 0.5 and row[N] == "1":
                nr_incorrect += 1
                for node in range(N):
                    # tweak weights:
                    update[node] = learning_rate * (1 - 0) * input_nodes[node]

                # np.add(weights, np.transpose(update))
                # weights = weights + np.reshape(update, newshape=(5,1))
            weights = weights + np.reshape(update, newshape=(N+1, 1))
            #print(update, weights)
            #print(update)

        print(iter, nr_incorrect)
        if nr_incorrect < 1:
            break
        nr_incorrect = 0

    for row in data:
        input_nodes = np.asarray([1] + row[0:N])
        print(row[0:N], sigmoid(np.matmul(input_nodes, weights)))

        #print(weights)

    print(weights)

