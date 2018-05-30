# NC-Final-Project

## Project Proposal
Neural networks and especially deep neural networks are currently one of the best performing machine learning models. These models consists of several layers of neurons that are connected with each other. Input information is fed forward through the layers until it reaches the output layer. In learning tasks, the weights of the connections between layers are learned. This is done by minimizing a cost function (error function). Currently, the back-propagation algorithm, which calculates the gradient of the loss function with respect to the weights in the network, is used for that \cite{schmidhuber2015deep}. However, using back-propagation introduces the vanishing gradient problem. This difficulty addresses the problem that the weight updates are proportional to the gradient of the loss function and was first described by Hochreiter in 1991 \cite{hochreiter1991untersuchungen}. To tackle this problem, many researchers adjusted the architecture of the network, for example, Hochreiter \& Schmidhuber introduced the Long Short-Term Memory (LSTM) as a new variant of a recurrent neural network in 1997 \cite{hochreiter1997long}. The LSTM model adds gates to the recurrent network to ignore irrelevant input and to remember important information longer. 
% section about momentum

In 1995, Kennedy and Eberhart presented the Particle Swarm Optimization algorithm, that is able to optimize non-linear functions based on swarm intelligence \cite{kennedy2011particle}. In their study, Kennedy \& Eberhart (1995) showed that the PSO algorithm performed just as efficient as the back-propagation algorithm and performed slightly better than the back-propagation algorithm in terms of classification accuracy on a data set representing electroencephalogram spike waveforms (89\% vs. 92\%). 

In our study, we would like to compare these two approaches on a bigger variety of data sets to see if the PSO algorithm will always outperform the back-propagation algorithm in terms of training time and classification accuracy. In addition, we could also include other algorithms for non-linear function optimization, like the ant colony algorithm.

We would implement both (all three) algorithms ourselves in python. Maybe we can use parts of the keras library, which offers an easy API for deep neural networks.

## What we have done so far
Implemented 2 different algorithms from this course in the domain of training MLP. For a MLP with one hidden layer, the standard EA and the standard PSO algorithm are implemented. Also both algorithms can be tested on the iris and MNIST data set. PSO was also implemented for a MLP with two hidden layers.

## What is left to do
* implement EA for MLP with 2 hidden layers
* maybe variations of the algorithms
* do the same with backprob, but with keras for a baseline
* calculate performance with 10 fold cross validation for every algorithm on every dataset.
