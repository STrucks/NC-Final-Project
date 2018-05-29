import numpy as np
import math
import matplotlib.pyplot as plt

def load_iris():
    data = [line.replace("\n", "").split(",") for line in open("iris_data.txt", 'r').readlines()]
    data1 = [[float(line[0]), float(line[1]), float(line[2]), float(line[3]), [1,0,0]] for line in data if
            line[4] == "Iris-setosa"]
    data1 += [[float(line[0]), float(line[1]), float(line[2]), float(line[3]), [0,1,0]] for line in data if
            line[4] == "Iris-versicolor"]
    data1 += [[float(line[0]), float(line[1]), float(line[2]), float(line[3]), [0,0,1]] for line in data if
            line[4] == "Iris-virginica"]
    np.random.shuffle(data1)
    return data1


def load_XOR():
    data = [line.replace("\n", "").split(",") for line in open("XOR.txt", 'r').readlines()]
    data1 = [[float(line[0]), float(line[1]), float(line[2])] for line in data]
    # np.random.shuffle(data1)
    return data1


def sigmoid(x):
    try:
        if hasattr(x, "__len__"): # input is a vector
            return [sigmoid(y) for y in x]
        else: # input is a scalar
            return 1 / (1 + math.exp(-x))
    except:
        if x > 0:
            return 1
        else:
            return 0


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def predict(input, hidden1, output):
    # hidden1 must be in format a x b with a being weights from input a to hidden b.
    # output weights must be in format b x c being the connections between the last hidden layer and the output
    hidden_units = sigmoid(np.dot(input, hidden1))
    #print("pred", input, hidden1, hidden_units)
    output_units = sigmoid(np.matmul(hidden_units, output))
    return np.asarray(hidden_units), output_units


def avg(matrix):
    avg_vec = np.zeros(shape=(len(matrix[1]), len(matrix[1][1])))
    for row in matrix:
        avg_vec += row
    return np.asarray(avg_vec/len(matrix))


def learn_weights_backprop(data, N, M, O):

    hidden_weights = 0.5 * (np.random.rand(N, M) - 0.5)
    hidden_bias = np.random.rand(M) - 0.5
    output_weights = np.random.rand(M, O) - 0.5
    output_bias = np.random.rand(O) - 0.5
    """
    hidden_weights = np.asarray([[-0.5,-1.5], [1,1],[-1,-1]])
    output_weights = np.asarray([[1],[1]])
    hu = np.matmul(np.asarray([1,1,0]), hidden_weights)
    ou = np.matmul(sigmoid(hu), sigmoid(output_weights))
    
    print("hidden", hu)
    print("out", ou)
    """
    for iter in range(10000):
        mistakes = 0
        output_update = []
        hidden_update = []
        for row in data:
            x = np.asarray(row[0:N])
            H, Y = predict(x, hidden_weights, output_weights)
            #print("Y", Y)

            mistakes += 1
            t = np.asarray(row[-1])
            output_error = Y * (np.ones(O) - Y) * (t - Y)

            #print(output_weights, output_error)
            # print(output_weights, output_error, np.dot(output_weights, np.transpose(output_error)))
            # np.dot(o, np.transpose(output_error)) * hidden_units * (np.ones(2) - hidden_units)
            #print(H, M)
            hidden_error = np.dot(output_weights, np.transpose(output_error)) * H * (np.ones(M) - H)

            # update the weights:
            hidden_update.append(eta * np.matmul(np.reshape(x, newshape=(N,1)), np.reshape(hidden_error, newshape=(1,M))))
            # print(hidden_update)
            output_update.append(eta * np.matmul(np.reshape(H, newshape=(M,1)), np.reshape(output_error, newshape=(1,O))))

            # prevent divergence:
            for i in range(len(hidden_update[-1])):
                for j in range(len(hidden_update[-1][i])):
                    if hidden_update[-1][i][j] > 10000:
                        hidden_update[-1][i][j] = 10000
                    if hidden_update[-1][i][j] < 0.00000000001:
                        hidden_update[-1][i][j] = 0
            for i in range(len(output_update[-1])):
                for j in range(len(output_update[-1][i])):
                    if output_update[-1][i][j] > 10000:
                        output_update[-1][i][j] = 10000
                    if output_update[-1][i][j] < 0.00000000001:
                        output_update[-1][i][j] = 0

        #print(avg(hidden_update))
        hidden_weights = hidden_weights + avg(hidden_update)
        output_weights = output_weights + avg(output_update)
        #print(iter, mistakes)
        #for row in data:
        #    print(row[0:N], predict(np.asarray([1] + row[0:N]), hidden_weights, output_weights))
    return hidden_weights, output_weights


def SSE(x, y):
    s = 0
    for i in range(len(x)):
        s += (x[i] - y[i]) ** 2
    return s

def learn_weights_PSO(data, N, M, O):
    POP_SIZE = 100
    MOMENTUM = 0.8
    ALPHA = [1,2]
    population = []

    # initialize the population:
    for p in range(POP_SIZE):
        population.append({
            'current': {
                'hidden': np.random.rand(N + 1, M)*10 - 5
                , 'output': np.random.rand(M, O)*100 - 50
            }
            , 'velocity': {
                'hidden': np.zeros(shape=(N + 1, M))
                , 'output': np.zeros(shape=(M, O))
            }
            ,'pbest': {
                'hidden': np.zeros(shape=(N + 1, M))
                , 'output': np.zeros(shape=(M, O))
            }
            , 'gbest': {
                'hidden': np.zeros(shape=(N + 1, M))
                , 'output': np.zeros(shape=(M, O))
            }
            , 'pbestScore': 99999
            , 'gbestScore': 99999
        })




    for iter in range(100):
        for p in population:
            # create the random vectors:
            r11 = np.random.rand(N + 1, M)
            r12 = np.random.rand(N + 1, M)
            r21 = np.random.rand(M, O)
            r22 = np.random.rand(M, O)

            # update the velocity:
            p['velocity']['hidden'] = MOMENTUM * p['velocity']['hidden'] + np.multiply(ALPHA[0] * r11, (
                        p['pbest']['hidden'] - p['current']['hidden'])) + np.multiply(ALPHA[1] * r12, (p['gbest']['hidden'] - p['current']['hidden']))
            p['velocity']['output'] = MOMENTUM * p['velocity']['output'] + np.multiply(ALPHA[0] * r21, (
                        p['pbest']['output'] - p['current']['output'])) + np.multiply(ALPHA[1] * r22, (p['gbest']['output'] - p['current']['output']))

            # update the position:
            p['current']['hidden'] += p['velocity']['hidden']
            p['current']['output'] += p['velocity']['output']

        # update local best
        for p in population:
            # to calculate the fitness, we will calculate the performance on the whole dataset:
            score = 0.
            for row in data:
                # H, Y = predict(np.asarray([1] + row[0:N]), p['current']['hidden'], p['current']['output'])
                X = np.asarray([1] + row[0:N])
                hidden_units = sigmoid(np.matmul(X, p['current']['hidden']))
                Y = sigmoid(np.matmul(hidden_units, p['current']['output']))
                score += SSE(Y, row[-1])
                # if list(Y).index(max(Y)) != row[-1].index(max(row[-1])): # the prediction was wrong
                #     score += 1
            if score <= p['pbestScore']:
                p['pbest']['hidden'] = p['current']['hidden']
                p['pbest']['output'] = p['current']['output']
                p['pbestScore'] = score
        # update global best:
        PBs = [p['pbestScore'] for p in population]
        GB = min(PBs)
        index_GB = PBs.index(GB)
        for p in population:
            p['gbestScore'] = GB
            p['gbest']['hidden'] = population[index_GB]['pbest']['hidden']
            p['gbest']['output'] = population[index_GB]['pbest']['output']

        print(iter, GB)

    return population[0]['gbest']['hidden'], population[0]['gbest']['output']


if __name__ == '__main__':
    data = load_XOR()
    N = 2  # size of input vector
    M = 3  # number of hidden layer units
    O = 1  # number of output units (classes)
    eta = 0.005  # learning rate
    # print(sum([[[1,1,5], [2,3,2]], [[1,1,5], [2,3,2]]]))
    print(predict([1,1], [[1,1,5], [2,3,2]], [[1,1],[2,2],[3,4]]))
    print("data", data)
    hidden_weights, output_weights = learn_weights_backprop(data, N, M, O)
    print(hidden_weights, output_weights)

    # see the performance by ploting the data + labels:
    for row in data:
        inp = np.asarray(row[0:N])
        H, Y = predict(inp, hidden_weights, output_weights)
        # label = list(Y).index(max(Y))
        print("y", Y)
        if Y[0] > 0:
            label = 1
        else:
            label = 0

        x1 = 0
        x2 = 1
        print(row[x1], row[x2])
        if label == 0:
            plt.plot(row[x1], row[x2], 'bx')
        if label == 1:
            plt.plot(row[x1], row[x2], 'rx')
        if label == 2:
            plt.plot(row[x1], row[x2], 'gx')

    plt.show()

    i = np.asarray([10,30,20])
    h = [[0.2,0.7],[-0.1,-1.2],[0.4,1.2]]
    o = [[1.1,3.1],[0.1,1.17]]
    print("1", np.matmul(i, h))
    hidden_units = sigmoid(np.matmul(i, h))
    output_units = sigmoid(np.matmul(hidden_units, o))
    output_error = output_units * (np.ones(2) - output_units) * (np.asarray([1,0]) - output_units)
    hidden_error = np.dot(o, np.transpose(output_error)) * hidden_units* (np.ones(2) - hidden_units)
    hidden_update = eta * np.matmul(np.reshape(i, newshape=(3, 1)), np.reshape(hidden_error, newshape=(1, 2)))
    print(hidden_units)
    print(np.matmul(np.reshape(i, newshape=(3, 1)), np.reshape(hidden_error, newshape=(1, 2))))
    print("out",hidden_update)


    #print(np.matmul([[1.1], [3.1]], [[2, 3]]))

    #print(output_weights)



