import nengo
import numpy as np

model = nengo.Network()
'''
with model:
    ensemblex = nengo.Ensemble(n_neurons=5, dimensions=1)
    #node_number = nengo.Node(output=0.5)
    node_function=nengo.Node(output=np.sin)
''' 
with model:
    ens = nengo.Ensemble(n_neurons=500, dimensions=1)
    #node_number = nengo.Node(output=0.5)
    node_function=nengo.Node(output=np.sin)
    nengo.Connection(node_function, ens)
    print(ens.probeable)

    

