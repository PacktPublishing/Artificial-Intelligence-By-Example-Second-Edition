# -*- coding: utf-8 -*-
# Decision Tree example for warehouse priority classification
# Copyright 2019 Denis Rothman MIT License. See LICENSE.
#import sklearn # for version check sklearn version 0.21.3
#print("sklearn version",sklearn.__version__) 


from sklearn import tree


# DECISION TREE LEARNING :DECISION TREE CLASSIFIER
#https://en.wikipedia.org/wiki/Decision_tree_learning
 
# 1. Data Collection created from the value of each O1 location in
#    the warehouse sample based on 3 features:
#  a) priority/location weight which bears a heavy weight to make a decison because of the cost of transporting distances
#  b) a volume priority weight which is set to 1 because in the weights were alrady measured to create reward matrix
#  c) high or low probablities determined by an optimization factor. For this example, distance

# 2.Providing the features of the dataset
features = [ 'Priority/location', 'Volume', 'Flow_optimizer' ]

Y = ['Low', 'Low', 'High', 'High', 'Low', 'Low']    

# 3. The data itself extracted from the result matrix
X = [ [256, 1,0],     
      [320, 1,0],
      [500, 1,1],
      [400, 1,0],
      [320, 1,0],
      [256, 1,0]]
 
# 4. Running the standard inbuilt tree classifier
classify = tree.DecisionTreeClassifier()
classify = classify.fit(X,Y)
print(classify)

# 5.Producing visualization(optional)
import collections       # from Python library container datatypes
import pydotplus         # a Python Interface to Graphviz’s Dot language.(dot-V command line for version) Graphviz version 2.40.1  https://pypi.org/project/pydotplus/

info = tree.export_graphviz(classify,feature_names=features,out_file=None,filled=True,rounded=True)
graph = pydotplus.graph_from_dot_data(info)
 
edges = collections.defaultdict(list) 
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))
 
for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0] 

graph.write_png('warehouse_example_decision_tree.png')
print("Open the image to verify that the priority level fits the reality of the reward matrix inputs")
