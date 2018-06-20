# NC-Final-Project

## How to use the code
to run the pso or the ES algorithm with one hidden layer, use mlp.py. If you want the model with two hidden layers, use mllp.py.
If you want to run the experiments, you have to adapt only a few lines: 

# load the data:
change line 436 to 
inputs, labels = load_iris()  OR
inputs, labels = load_artificial_ds2()
to load either the iris data set or the artificial data set in the mlp.py. 

Same for mllp.py, but line 471

Note: make sure to adapt the input layer. The iris data set uses 4 input nodes while the artificial data set uses only 2. To change that, alter the value of I in the main function to either 2 or 4.


# PSO algorithm
if you want to run the pso algorithm make sure that line 286 in mlp.py is

return self.train_pso(data, training_set_outputs, number_of_training_iterations=1000)

same for mllp.py but then line 356. Then just hit run.

# ES algorithm
if you want to run the ES algorithm make sure that line 286 in mlp.py is

return self.train_EA(data, training_set_outputs, number_of_training_iterations=1000)

same for mllp.py but then line 356. Then just hit run.

# backprop
if you want to run the BP algorithm, just execute backprop.py or backprop_artificial.py. backprop uses the iris data set while backprop_artificial uses the artificial data set. Set the variable NUM_HIDDEN_LAYERS to 1 or 2 depending on how many hidden layers you want.


