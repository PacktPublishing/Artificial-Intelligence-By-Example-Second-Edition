#Cognitive NPL (Natural Language Processing)
#Copyright 2020 Denis Rothman MIT License. READ LICENSE.
#Personality Profiling with a Restricted Botzmannm Machine (RBM)

import numpy as np
from random import randint

class RBM:
  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.debug_print = True

    # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    # a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
    # and sqrt(6. / (num_hidden + num_visible)).
    # Standard initialization  the weights with mean 0 and standard deviation 0.1. 
    #Starts with random state 
    np_rng = np.random.RandomState(1234)
    self.weights = np.asarray(np_rng.uniform(
			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	size=(num_visible, num_hidden)))


    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self, data, max_epochs, learning_rate):

    num_examples = data.shape[0]

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs):      
      # Linking the data and sample from the hidden units. 
      # (This is the "positive CD phase", aka the reality phase.)
      pos_hidden_activations = np.dot(data, self.weights)      
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_probs[:,0] = 1 # Fix the bias unit.
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      pos_associations = np.dot(data.T, pos_hidden_probs)
      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
      # Update weights.
      self.weights += learning_rate * ((pos_associations - neg_associations))

      error = np.sum((data - neg_visible_probs) ** 2)
      energy=-np.sum(data) - np.sum(neg_hidden_probs)-np.sum(pos_associations * self.weights)
      z=np.sum(data)+np.sum(neg_hidden_probs)
      if z>0: energy=np.exp(-energy)/z;
      if self.debug_print:
        print("Epoch %s: error is %s" % (epoch, error)," Energy:",energy)    
      
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
  r = RBM(num_visible = 6, num_hidden = 2)
  training_data = np.array([[1,1,0,0,1,1],
                            [1,1,0,1,1,0],
                            [1,1,1,0,0,1],
                            [1,1,0,1,1,0],
                            [1,1,0,0,1,0],
                            [1,1,1,0,1,0]])

  F=["love","happiness","family","horizons","action","violence"]
  print(" A Restricted Boltzmann Machine(RBM)","\n","applied to profiling a person name X","\n","based on the movie ratings of X.","\n")
  print("The input data represents the features to be trained to learn about person X.")
  print("\n","Each colum represents a feature of X's potential pesonality and tastes.")
  print(F,"\n")
  print(" Each line is a movie X watched containing those 6 features")
  print(" and for which X gave a 5 star rating.","\n")
  print(training_data)
  print("\n")
  max_epochs=5000
  learning_rate = 0.001
  r.train(training_data, max_epochs,learning_rate)
  print("\n","The weights of the features have been trained for person X.","\n","The first line is the bias and examine column 2 and 3","\n","The following 6 lines are X's features.","\n")
  print("Weights:")
  print(r.weights)
  print("\n","The following array is a reminder of the features of X.")
  print(" The columns are the potential features of X.","\n", "The lines are the movies highly rated by X")
  print(F,"\n")
  print(training_data)
  print("\n")
  print("The results are only experimental results.","\n")
  
  for w in range(7):
      if(w>0):
          W=print(F[w-1],":",r.weights[w,1]+r.weights[w,2])
          
  print("\n")
  print("A value>0 is positive, close to 0 slightly positive")
  print("A value<0 is negative, close to 0 slightly negative","\n")
  
   
