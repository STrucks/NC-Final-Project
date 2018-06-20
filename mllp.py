from numpy import exp, array, random, dot, zeros, multiply, transpose, exp, max, ones, reshape, std
import matplotlib.pyplot as plt

"""
This file is pretty much the same as mlp.py, but here we will use 2 hidden layers. I created a second file for this 
to not lose my progress.
"""


def deep_copy( a):
    b = zeros(shape=(len(a), len(a[0])))
    for i in range(len(a)):
        for j in range(len(a[0])):
            b[i][j] = a[i][j]
    return b

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.number_of_neurons = number_of_neurons
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2, layer3):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (ones(shape=(len(x), len(x[0]))) - x)

    def __softmax(self, x):
        Y = []
        for row in x:
            Y.append(exp(row)/sum(exp(row)))
        return Y

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train_backprop(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = array(training_set_outputs) - array(output_from_layer_2)
            print(output_from_layer_2, layer2_error)
            layer2_delta = multiply(layer2_error , self.__sigmoid_derivative(output_from_layer_2))

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    def deep_copy(self, a):
        b = zeros(shape=(len(a), len(a[0])))
        for i in range(len(a)):
            for j in range(len(a[0])):
                b[i][j] = a[i][j]
        return b

    def SSE(self, x, y):
        s = 0
        for i in range(len(x)):
            s += (x[i] - y[i]) ** 2
        return sum(s)

    def train_pso(self, training_set_inputs, training_set_outputs, number_of_training_iterations = 50):
        POP_SIZE = 30
        MOMENTUM = 0.8
        ALPHA = [0.9, 1.2]
        reset_chance = 0.01
        population = []
        scores = []
        # initialize the population:
        for p in range(POP_SIZE):
            hidden = NeuronLayer(I, H)
            hidden2 = NeuronLayer(H, H2)
            output = NeuronLayer(H2, O)
            population.append({
                'current': {
                    'hidden':       hidden,
                    'hidden2':      hidden2,
                    'output':       output
                }
                , 'velocity': {
                    'hidden':       zeros(shape=(I, H)),
                    'hidden2':       zeros(shape=(H, H2)),
                    'output':       zeros(shape=(H2, O))
                }
                , 'pbest': {
                    'hidden':       NeuronLayer(I, H),
                    'hidden2':      NeuronLayer(H, H2),
                    'output':       NeuronLayer(H2, O)
                }
                , 'gbest': {
                    'hidden':       NeuronLayer(1, 1),
                    'hidden2':      NeuronLayer(H, H2),
                    'output':       NeuronLayer(1, 1)
                }
                , 'pbestScore':     99999
                , 'gbestScore':     99999
            })
            population[-1]['pbest']['hidden'].synaptic_weights = self.deep_copy(hidden.synaptic_weights)
            population[-1]['pbest']['hidden2'].synaptic_weights = self.deep_copy(hidden2.synaptic_weights)
            population[-1]['pbest']['output'].synaptic_weights = self.deep_copy(output.synaptic_weights)

        # find global best:
        PBs = [p['pbestScore'] for p in population]
        GB = min(PBs)
        index_GB = PBs.index(GB)
        for p in population:
            p['gbestScore'] = GB
            p['gbest']['hidden'].synaptic_weights = population[index_GB]['pbest']['hidden'].synaptic_weights
            p['gbest']['output'].synaptic_weights = population[index_GB]['pbest']['output'].synaptic_weights

        for iteration in range(number_of_training_iterations):
            for p in population:
                if random.randint(0, 1000) < reset_chance * 1000:
                    # reset the current position of the particle
                    p['hidden'] = NeuronLayer(I, H)
                    p['hidden2'] = NeuronLayer(H, H2)
                    p['output'] = NeuronLayer(H2, O)

                else:
                    # create the random vectors:
                    r11 = transpose(random.rand(I, H))
                    r12 = transpose(random.rand(I, H))
                    r21 = transpose(random.rand(H, H2))
                    r22 = transpose(random.rand(H, H2))
                    r31 = transpose(random.rand(H2, O))
                    r32 = transpose(random.rand(H2, O))
                    # update the velocity:
                    p['velocity']['hidden'] = MOMENTUM * p['velocity']['hidden'] + transpose(\
                                              + ALPHA[0] * r11 * (p['pbest']['hidden'].synaptic_weights - p['current']['hidden'].synaptic_weights) \
                                              + ALPHA[1] * r12 * (p['gbest']['hidden'].synaptic_weights - p['current']['hidden'].synaptic_weights))
                    p['velocity']['hidden2'] = MOMENTUM * p['velocity']['hidden2'] + transpose(\
                                              + ALPHA[0] * r21 * (p['pbest']['hidden2'].synaptic_weights - p['current']['hidden2'].synaptic_weights) \
                                              + ALPHA[1] * r22 * (p['gbest']['hidden2'].synaptic_weights - p['current']['hidden2'].synaptic_weights))
                    p['velocity']['output'] = MOMENTUM * p['velocity']['output'] + transpose(\
                                              + ALPHA[0] * r31 * (p['pbest']['output'].synaptic_weights - p['current']['output'].synaptic_weights) \
                                              + ALPHA[1] * r32 * (p['gbest']['output'].synaptic_weights - p['current']['output'].synaptic_weights))

                    # update the position:
                    p['current']['hidden'].synaptic_weights += transpose(p['velocity']['hidden'])
                    p['current']['hidden2'].synaptic_weights += transpose(p['velocity']['hidden2'])
                    p['current']['output'].synaptic_weights += transpose(p['velocity']['output'])

            # update local best
            for p in population:
                # to calculate the fitness, we will calculate the performance on the whole dataset:
                output_from_layer1 = self.__sigmoid(dot(training_set_inputs, transpose(p['current']['hidden'].synaptic_weights)))
                output_from_layer1_1 = self.__sigmoid(dot(output_from_layer1, transpose(p['current']['hidden2'].synaptic_weights)))
                output_from_layer_2 = self.__sigmoid(dot(output_from_layer1_1, transpose(p['current']['output'].synaptic_weights)))
                #print(self.__softmax(output_from_layer_2), training_set_outputs)

                score = self.SSE(self.__softmax(output_from_layer_2), training_set_outputs)
                #print(score)
                if score <= p['pbestScore']:
                    p['pbest']['hidden'].synaptic_weights = self.deep_copy(p['current']['hidden'].synaptic_weights)
                    p['pbest']['hidden2'].synaptic_weights = self.deep_copy(p['current']['hidden2'].synaptic_weights)
                    p['pbest']['output'].synaptic_weights = self.deep_copy(p['current']['output'].synaptic_weights)
                    p['pbestScore'] = score

            # update global best:
            PBs = [p['pbestScore'] for p in population]
            GB = min(PBs)
            index_GB = PBs.index(GB)
            for p in population:
                p['gbestScore'] = GB
                p['gbest']['hidden'] = population[index_GB]['pbest']['hidden']
                p['gbest']['hidden2'] = population[index_GB]['pbest']['hidden2']
                p['gbest']['output'] = population[index_GB]['pbest']['output']
            scores.append(GB)
            #print(iteration, GB)
            if len(scores) > 50:
                if scores[-1] == scores[-50]:
                    print("no improvement after", iteration)
                    break
            # MOMENTUM -= 1/number_of_training_iterations
        self.layer1.synaptic_weights = transpose(population[0]['gbest']['hidden'].synaptic_weights)
        self.layer2.synaptic_weights = transpose(population[0]['gbest']['hidden2'].synaptic_weights)
        self.layer3.synaptic_weights = transpose(population[0]['gbest']['output'].synaptic_weights)
        return self

    def train_EA(self, training_set_inputs, training_set_outputs, number_of_training_iterations=50):
        POP_SIZE = 30
        MUTATION_RATE = 0.01
        population = []
        mask_hidden = reshape([random.choice([0, 1]) for i in range(I * H)], newshape=(H, I))
        mask_hidden2 = reshape([random.choice([0, 1]) for i in range(H2 * H)], newshape=(H2, H))
        mask_output = reshape([random.choice([0,1]) for i in range(H2*O)], newshape=(O,H2))

        # initialize the population:
        for p in range(POP_SIZE):
            hidden = NeuronLayer(I, H)
            hidden2 = NeuronLayer(H, H2)
            output = NeuronLayer(H2, O)
            population.append({
                    'hidden':   hidden,
                    'hidden2':  hidden2,
                    'output':   output,
                    'fitness':  0,

            })
        # evaluate every guy:
        fitnesses = []
        for p in population:
            output_from_layer1 = self.__sigmoid(
                dot(training_set_inputs, transpose(p['hidden'].synaptic_weights)))
            output_from_layer1_1 = self.__sigmoid(
                dot(output_from_layer1, transpose(p['hidden2'].synaptic_weights)))
            output_from_layer_2 = self.__sigmoid(
                dot(output_from_layer1_1, transpose(p['output'].synaptic_weights)))

            p['fitness'] = -(self.SSE(output_from_layer_2, training_set_outputs) + 1)
            fitnesses.append(p['fitness'])
        #print("best", max(fitnesses), std(fitnesses))
        scores = []
        for iter in range(number_of_training_iterations):
            # select a mating pool by roulette wheel selection:
            selection = []
            all_fit = sum([p['fitness'] for p in population])
            #print(all_fit)
            for i in range(POP_SIZE):
                t = random.random() * all_fit
                #print("t", t)
                for p in population:
                    t -= p['fitness']
                    if t < 0:
                        h = NeuronLayer(I, H)
                        h.synaptic_weights = self.deep_copy(p['hidden'].synaptic_weights)
                        h2 = NeuronLayer(H, H2)
                        h2.synaptic_weights = self.deep_copy(p['hidden2'].synaptic_weights)
                        o = NeuronLayer(H2, O)
                        o.synaptic_weights = self.deep_copy(p['output'].synaptic_weights)
                        selection.append({
                            'hidden':       h,
                            'hidden2':      h2,
                            'output':       o,
                            'fitness':      p['fitness'],

                        })
                        break
            # create a new population with cross over (binary mask)
            new_pop = []
            for i in range(POP_SIZE):
                p1 = random.choice(population)
                p2 = random.choice(population)
                new_guy = {
                    'hidden':       NeuronLayer(I,H),
                    'hidden2':      NeuronLayer(H,H2),
                    'output':       NeuronLayer(H2,O),
                    'fitness':      0,

                }
                for i in range(H):
                    for j in range(I):
                        if mask_hidden[i][j] == 0:
                            new_guy['hidden'].synaptic_weights[i][j] = p1['hidden'].synaptic_weights[i][j]
                        else:
                            new_guy['hidden'].synaptic_weights[i][j] = p2['hidden'].synaptic_weights[i][j]
                for i in range(H2):
                    for j in range(H):
                        if mask_hidden2[i][j] == 0:
                            new_guy['hidden2'].synaptic_weights[i][j] = p1['hidden2'].synaptic_weights[i][j]
                        else:
                            new_guy['hidden2'].synaptic_weights[i][j] = p2['hidden2'].synaptic_weights[i][j]

                for i in range(O):
                    for j in range(H2):
                        if mask_output[i][j] == 0:
                            new_guy['output'].synaptic_weights[i][j] = p1['output'].synaptic_weights[i][j]
                        else:
                            new_guy['output'].synaptic_weights[i][j] = p2['output'].synaptic_weights[i][j]

                new_pop += [new_guy]

            # do mutation:
            for p in population:
                if random.randint(0,1000) < MUTATION_RATE * 1000:
                    p['hidden'].synaptic_weights += reshape([random.normal(scale=1) for i in range(I * H)], newshape=(H,I))
                    p['hidden2'].synaptic_weights += reshape([random.normal(scale=1) for i in range(H2 * H)], newshape=(H2, H))
                    p['output'].synaptic_weights += reshape([random.normal(scale=1) for i in range(H2 * O)], newshape=(O, H2))

            # evaluate new generation:
            fitnesses = []
            for p in new_pop:
                output_from_layer1 = self.__sigmoid(
                    dot(training_set_inputs, transpose(p['hidden'].synaptic_weights)))
                output_from_layer1_1 = self.__sigmoid(
                    dot(output_from_layer1, transpose(p['hidden2'].synaptic_weights)))
                output_from_layer_2 = self.__sigmoid(
                    dot(output_from_layer1_1, transpose(p['output'].synaptic_weights)))

                p['fitness'] = -(self.SSE(output_from_layer_2, training_set_outputs) + 1)
                fitnesses.append(p['fitness'])

            # combine both populations and remove the weaker half:
            population += new_pop
            population = list(reversed(sorted(population, key=lambda k: k['fitness'])))[0:POP_SIZE]

            #print(iter, population[0]['fitness'])
        self.layer1.synaptic_weights = transpose(population[0]['hidden'].synaptic_weights)
        self.layer2.synaptic_weights = transpose(population[0]['hidden2'].synaptic_weights)
        self.layer3.synaptic_weights = transpose(population[0]['output'].synaptic_weights)

    def get_performance(self, input_data, labels):
        perf = 0
        for index, row in enumerate(input_data):
            output_from_layer1, output_from_layer1_1, output_from_layer2 = self.think(row)
            training_set_outputs = [0]*len(output_from_layer2)
            training_set_outputs[labels[index]] = 1
            perf += self.SSE(output_from_layer2, training_set_outputs)
        return perf

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer1_1 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1_1, self.layer3.synaptic_weights))
        return output_from_layer1, output_from_layer1_1, self.__softmax(output_from_layer2)

    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 1 (", str(H), "neurons, each with", str(I), "inputs): ")
        print(self.layer1.synaptic_weights)
        print("    Layer 2 (", str(O), "neuron, with", str(H), "inputs):")
        print(self.layer2.synaptic_weights)

    def fit(self, data, labels):
        self.reset()
        Y = array(labels).T
        training_set_outputs = []
        for y in Y:
            training_set_outputs.append([0] * O)
            training_set_outputs[-1][y] = 1
        return self.train_EA(data, training_set_outputs, number_of_training_iterations=1000)

    def predict(self, row):
        h1, h2, output = self.think(row)
        return [list(o).index(max(o)) for o in output]

    def reset(self):
        self.layer1 = NeuronLayer(H, I)
        self.layer2 = NeuronLayer(H2, H)
        self.layer3 = NeuronLayer(O, H2)


def load_iris():
    raw_data = [line.replace("\n", "").split(",") for line in open("iris_data.txt", 'r').readlines()]
    data = []
    labels = []
    for line in raw_data:
        data.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
        if line[-1] == "Iris-setosa":
            labels.append(0)
        elif line[-1] == "Iris-versicolor":
            labels.append(1)
        else:
            labels.append(2)
    return data, labels


def load_XOR():
    data = [line.replace("\n", "").split(",") for line in open("XOR.txt", 'r').readlines()]
    data1 = [[float(line[0]), float(line[1])] for line in data]
    labels = [float(line[2]) for line in data]
    # np.random.shuffle(data1)
    return data1, labels


def load_MNIST():
    from sklearn.datasets import load_digits
    digits = load_digits()
    data = [reshape(img, newshape=(1,8*8)) for img in digits['images']]
    data = [d[0] for d in data]
    labels = digits['target']
    return data, labels


def load_wine():
    from sklearn.datasets import load_wine
    data = load_wine()

    return data['data'], data['target']


def load_artificial_ds1():
    import math
    data = random.rand(200,2) * 4
    labels = []
    for x, y in data:
        if math.sqrt(x ** 2 + y ** 2) < 2:
            labels.append(0)
        elif math.sqrt(x ** 2 + y ** 2) < 3:
            labels.append(1)
        elif math.sqrt(x ** 2 + y ** 2) < 4:
            labels.append(2)
        else:
            labels.append(3)
    plot = False
    if plot:
        import matplotlib.pyplot as plt
        colors = ['b', 'r', 'g', 'y']
        for row, label in zip(data, labels):
            plt.plot(row[0], row[1], colors[label] + '.')
        plt.show()
    return data, labels


def load_artificial_ds2():
    import math
    data = random.rand(500, 2) * 8 - 4
    labels = []
    for x, y in data:
        if math.sqrt(x ** 2 + y ** 2) < 2:
            labels.append(0)
        elif math.sqrt(x ** 2 + y ** 2) < 3:
            labels.append(0)
        elif math.sqrt(x ** 2 + y ** 2) < 4:
            labels.append(1)
        else:
            labels.append(2)
    plot = False
    if plot:
        import matplotlib.pyplot as plt
        colors = ['b', 'r', 'g', 'y']
        for row, label in zip(data, labels):
            plt.plot(row[0], row[1], colors[label] + '.')
        plt.show()
    return data, labels


def run_MLP_CV():
    # Seed the random number generator
    # random.seed(1)
    # Create layer 1 (4 neurons, each with 3 inputs)
    layer1 = NeuronLayer(H, I)
    layer2 = NeuronLayer(H2, H)

    # Create layer 2 (a single neuron with 4 inputs)
    layer3 = NeuronLayer(O, H2)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2, layer3)

    #print("Stage 1) Random starting synaptic weights: ")
    # neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    inputs, labels = load_artificial_ds2()
    training_set_inputs = array(inputs)
    Y = array(labels).T
    training_set_outputs = []
    for y in Y:
        training_set_outputs.append([0] * O)
        training_set_outputs[-1][y] = 1

    # do CV:
    from cross_validation import CV
    CV(neural_network, training_set_inputs, labels, nr_folds=10)


def fit_on_data():
    # Seed the random number generator
    # random.seed(1)
    # Create layer 1 (4 neurons, each with 3 inputs)
    layer1 = NeuronLayer(H, I)
    layer2 = NeuronLayer(H2, H)

    # Create layer 2 (a single neuron with 4 inputs)
    layer3 = NeuronLayer(O, H2)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2, layer3)

    print("Stage 1) Random starting synaptic weights: ")
    # neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    inputs, labels = load_artificial_ds2()
    training_set_inputs = array(inputs)
    Y = array(labels).T
    training_set_outputs = []
    for y in Y:
        training_set_outputs.append([0] * O)
        training_set_outputs[-1][y] = 1

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train_pso(training_set_inputs, training_set_outputs, 1000)


    # Test the neural network with all training data
    hidden_state, hidden_state2, output = neural_network.think(training_set_inputs)
    plt.figure(1)
    nr_misclassifications = 0
    # see the performance by ploting the data + labels:
    plot = True
    for index, row in enumerate(output):
        print(index, row)
        if labels[index] != list(row).index(max(row)):
            nr_misclassifications += 1

        if plot:
            x2 = 1
            x1 = 0
            x = training_set_inputs[index][x1]
            y = training_set_inputs[index][x2]
            if list(row).index(max(row)) == 0:
                plt.plot(x, y, 'bo')
            elif list(row).index(max(row)) == 1:
                plt.plot(x,y, 'go')
            elif list(row).index(max(row)) == 2:
                plt.plot(x, y, 'yo')
            elif list(row).index(max(row)) == 3:
                plt.plot(x, y, 'o', color="GRAY")
            if labels[index] != list(row).index(max(row)):
                plt.plot(x, y, 'rx')

    print("number misclassifications:", nr_misclassifications)
    if plot:
        print("show")
        plt.show()


def find_params():
    I = 4
    H = 10
    H2 = 10
    O = 3
    inputs, labels = load_artificial_ds2()
    training_set_inputs = array(inputs)
    Y = array(labels).T
    training_set_outputs = []
    for y in Y:
        training_set_outputs.append([0] * O)
        training_set_outputs[-1][y] = 1
    # we can only vary on H and H2:
    POP_SIZE = 20
    MUTATION_RATE = 0.01
    population = []
    for i in range(POP_SIZE):
        H = random.randint(2,50)
        H2 = random.randint(2,50)
        layer1 = NeuronLayer(H, I)
        layer2 = NeuronLayer(H2, H)
        layer3 = NeuronLayer(O, H2)
        neural_network = NeuralNetwork(layer1, layer2, layer3)
        population.append(neural_network)
    # evaluate every guy:
    fitnesses = []
    for p in population:
        p.train_pso(training_set_inputs, training_set_outputs, 20)
        fitnesses.append(p.get_performance(training_set_inputs, labels))

    scores = []
    for iter in range(50):
        # select a mating pool by roulette wheel selection:
        selection = []
        all_fit = sum(fitnesses)
        # print(all_fit)
        for i in range(POP_SIZE):
            t = random.random() * all_fit
            # print("t", t)
            for i, f in enumerate(fitnesses):
                t -= f
                if t < 0:
                    h = NeuronLayer(I, population[i].layer2.number_of_neurons)
                    h.synaptic_weights = deep_copy(population[i].layer1.synaptic_weights)
                    h2 = NeuronLayer(population[i].layer2.number_of_neurons, population[i].layer3.number_of_neurons)
                    h2.synaptic_weights = deep_copy(population[i].layer2.synaptic_weights)
                    o = NeuronLayer(population[i].layer3.number_of_neurons, O)
                    o.synaptic_weights = deep_copy(population[i].layer3.synaptic_weights)
                    selection.append(NeuralNetwork(h, h2, o))
                    break
        """
        # create a new population with cross over (binary mask)
        new_pop = []
        for i in range(POP_SIZE):
            p1 = random.choice(population)
            p2 = random.choice(population)
            new_guy = {
                'hidden': NeuronLayer(I, H),
                'hidden2': NeuronLayer(H, H2),
                'output': NeuronLayer(H2, O),
                'fitness': 0,

            }
            for i in range(H):
                for j in range(I):
                    if mask_hidden[i][j] == 0:
                        new_guy['hidden'].synaptic_weights[i][j] = p1['hidden'].synaptic_weights[i][j]
                    else:
                        new_guy['hidden'].synaptic_weights[i][j] = p2['hidden'].synaptic_weights[i][j]
            for i in range(H2):
                for j in range(H):
                    if mask_hidden2[i][j] == 0:
                        new_guy['hidden2'].synaptic_weights[i][j] = p1['hidden2'].synaptic_weights[i][j]
                    else:
                        new_guy['hidden2'].synaptic_weights[i][j] = p2['hidden2'].synaptic_weights[i][j]

            for i in range(O):
                for j in range(H2):
                    if mask_output[i][j] == 0:
                        new_guy['output'].synaptic_weights[i][j] = p1['output'].synaptic_weights[i][j]
                    else:
                        new_guy['output'].synaptic_weights[i][j] = p2['output'].synaptic_weights[i][j]

            new_pop += [new_guy]

        # do mutation:
        for p in population:
            if random.randint(0, 1000) < MUTATION_RATE * 1000:
                p['hidden'].synaptic_weights += reshape([random.normal(scale=1) for i in range(I * H)], newshape=(H, I))
                p['hidden2'].synaptic_weights += reshape([random.normal(scale=1) for i in range(H2 * H)],
                                                         newshape=(H2, H))
                p['output'].synaptic_weights += reshape([random.normal(scale=1) for i in range(H2 * O)],
                                                        newshape=(O, H2))

        # evaluate new generation:
        fitnesses = []
        for p in new_pop:
            output_from_layer1 = self.__sigmoid(
                dot(training_set_inputs, transpose(p['hidden'].synaptic_weights)))
            output_from_layer1_1 = self.__sigmoid(
                dot(output_from_layer1, transpose(p['hidden2'].synaptic_weights)))
            output_from_layer_2 = self.__sigmoid(
                dot(output_from_layer1_1, transpose(p['output'].synaptic_weights)))

            p['fitness'] = -(self.SSE(output_from_layer_2, training_set_outputs) + 1)
            fitnesses.append(p['fitness'])

        # combine both populations and remove the weaker half:
        population += new_pop
        population = list(reversed(sorted(population, key=lambda k: k['fitness'])))[0:POP_SIZE]
        """
        print(iter, population[0]['fitness'])

I = 0
H = 0
H2 = 0
O = 0

if __name__ == "__main__":

    I = 2
    H = 5
    H2 = 5
    O = 3
    run_MLP_CV()
    I = 2
    H = 10
    H2 = 10
    O = 3
    run_MLP_CV()
    I = 2
    H = 50
    H2 = 50
    O = 3
    run_MLP_CV()
    I = 2
    H = 100
    H2 = 100
    O = 3
    run_MLP_CV()





