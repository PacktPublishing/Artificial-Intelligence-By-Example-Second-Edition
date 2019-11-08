#Cognitive NPL (Natural Language Processing)
#Copyright 2020 Denis Rothman MIT License. READ LICENSE.
#Personality Profiling with a Restricted Botzmannm Machine (RBM)

import numpy as np
from random import randint

# Part II Restricted Boltzmann Machine
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
      # Clamp to the data and sample from the hidden units. 
      # (This is the "positive CD phase", aka the reality phase.)
      pos_hidden_activations = np.dot(data, self.weights)      
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_probs[:,0] = 1 # Fix the bias unit.
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      pos_associations = np.dot(data.T, pos_hidden_probs)
      # Reconstruct the visible units and sample again from the hidden units.
      # (This is the "negative CD phase", aka the daydreaming phase.)
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
      # Update weights.
      self.weights += learning_rate * ((pos_associations - neg_associations))

      error = np.sum((data - neg_visible_probs) ** 2)
      #if self.debug_print and epoch>me-2:
        #print("Epoch %s: error is %s" % (epoch, error));     
      
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

def main():
  # RBM_launcher option
  pt=0  #restricted printing(0), printing(1)

  # Part I Feature extractions from data sources
  # The titles of 10 movies
  titles=["24H in Kamba","Lost","Cube Adventures","A Holiday","Jonathan Brooks","The Melbourne File", "WNC Detectives","Stars","Space L","Zone 77"]
  # The feature map of each of the 10 movies. Each line is a movie.
  # Each column is a feature. There are 6 features: ['love', 'happiness', 'family', 'horizons', 'action', 'violence']
  # 1= the feature is activated, 0= the feature is not activated
  movies_feature_map = np.array([[1,1,0,0,1,1],
                                 [1,1,0,1,1,1],
                                 [1,0,0,0,0,1],
                                 [1,1,0,1,1,1],
                                 [1,0,0,0,1,1],
                                 [1,1,0,1,1,0],
                                 [1,0,0,0,0,0],
                                 [1,1,0,1,1,0],
                                 [1,1,0,0,0,1],
                                 [1,0,0,1,1,1],
                                 [1,1,0,0,1,0],
                                 [1,1,0,1,1,1],
                                 [1,1,0,0,1,1]])

  #The output matrix is empty before the beginning of the analysis
  #The program will take the user "likes" of 6 out of the 10 movies

  dialog_output = np.array([[0,0,0,0,0,0],
                           [0,0,0,0,0,0],
                           [0,0,0,0,0,0],
                           [0,0,0,0,0,0],
                           [0,0,0,0,0,0],
                           [0,0,0,0,0,0]])

  #An extraction of viewer's first 6 liked 6 movies out n choices
  #Hundreds of movies can be added. No dialog is needed since a cloud streaming services stores the movie-likes we click on
  mc=0   #Number of choices limited to 6 in this example
  a="no" #default input value if rd==1
  #for m in range(0,10):
  if pt==1:print("Movie likes:");
  while mc<6:
    m=randint(0,9)# filter a chosen movie or allow (this case) a person can watch and like a movie twice=an indication
    b=randint(0,1)# a person can like(dislike) a movie the first time and not the second(or more) time
    if mc<6 and (a=="yes" or b==1):
      if pt==1:print("title likes: ",titles[m]);
      for i in range(0,6): dialog_output[mc,i]=movies_feature_map[m,i];
      mc+=1
    if mc>=6:
      break

  #The dialog_input is now complete
  if pt==1:print("dialog output",dialog_output);

  #dialog_output= the training data
  training_data=dialog_output
  r = RBM(num_visible = 6, num_hidden = 2)
  max_epochs=5000
  learning_rate=0.001
  r.train(training_data, max_epochs,learning_rate)

###Processing the results
  # feature labels
  F=["love","happiness","family","horizons","action","violence"]
  #saving features and finding primary feature
  f= open("features.tsv","a")

  best1=-1000
  pos=10
  control=0
  if(control==1):
    tcont=0  
    control=[0,0,0,0,0,0]
    
    for j in range(0,6):
      for i in range(0,6):
        control[i]+=dialog_output[j][i]
        tcont+=dialog_output[j][i]

    tcontrol=control
    
    cf1=0
    for b in range(0,6):
      if tcontrol[b]>=tcontrol[cf1]:
        cf1=b

    first=0
    for b in range(0,6):
      if(b!=cf1):
        if(first==0):cf2=b;first=1;
        if(first>0):
          if tcontrol[b]>=tcontrol[cf2]:
            cf2=b

    totw=0
    #total rescaled weights
    for w in range(1,7):
        if(w>0):
            print(F[w-1],":",r.weights[w,0])
            per=r.weights[w,0]+pos
            totw+=per

    if(totw<=0):totw=1;
    
    #proportion of each feature
    for w in range(1,7):
        if(w>0):
          per=r.weights[w,0]+pos
          print("CONTROL: ",F[w-1],":",round(r.weights[w,0],3),totw,round(per/totw,3));

    control=[0,0,0,0,0,0]
    for j in range(0,6):
      for i in range(0,6):
        control[i]+=dialog_output[j][i]
             
  ###End of processing the results

  #Selection of the primary feature
  for w in range(1,7):
      if(w>0):
          if pt==1:print(F[w-1],":",r.weights[w,0]);
          tw=r.weights[w,0]+pos
          if(tw>best1):
             f1=w-1
             best1=tw
          f.write(str(r.weights[w,0]+pos)+"\t")
  f.write("\n")
  f.close()

  #secondary feature
  best2=-1000
  for w in range(1,7):
    if(w>0):
      tw=r.weights[w,0]+pos
      if(tw>best2 and w-1!=f1):
        f2=w-1
        best2=tw
  
  #saving the metadata with the labels
  u=randint(1,10000)
  vname="viewer_"+str(u)
  if(pt==1):
    print("Control",control)
    print("Principal Features: ",vname,f1,f2,"control")
  
  f= open("labels.tsv","a")
  f.write(vname +"\t"+F[f1]+"\t"+F[f2]+"\t")
  f.write("\n")
  f.close()
 
if __name__ == '__main__':
  main()
