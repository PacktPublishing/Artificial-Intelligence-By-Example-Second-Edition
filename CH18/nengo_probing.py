#https://www.nengo.ai/nengo/examples/basic/single_neuron.html
#Transformed by Denis Rothman for educational purposes
import nengo
import numpy as np

import matplotlib.pyplot as plt
from nengo.utils.matplotlib import rasterplot
from nengo.dists import Uniform

model = nengo.Network("Probing")
with model:
    ens = nengo.Ensemble(n_neurons=50, dimensions=1)
    #node_number = nengo.Node(output=0.5)
    node_function=nengo.Node(output=np.sin)
    nengo.Connection(node_function, ens)
    print(ens.probeable)

with model:
    # Connect the input signal to the neuron
    nengo.Connection(node_function, ens)
    
with model:
    # The original input
    function_probe = nengo.Probe(node_function)
    # The raw spikes from the neuron
    spikes = nengo.Probe(ens.neurons)
    # Subthreshold soma voltage of the neuron
    voltage = nengo.Probe(ens.neurons, 'voltage')
    # Spikes filtered by a 10ms post-synaptic filter
    filtered = nengo.Probe(ens, synapse=0.01)
    
with nengo.Simulator(model) as sim:  # Create the simulator
    sim.run(5) 


print("Decoded output of the ensemble")
print(sim.trange(), sim.data[filtered])

print("Spikes")
print(sim.trange(),sim.data[spikes])


print("Voltage")
print((sim.trange(), sim.data[voltage][:, 0]))


# Plot the decoded output of the ensemble
plt.figure()
plt.plot(sim.trange(), sim.data[filtered])
#plt.plot(sim.trange(), sim.data[node_function])
plt.xlim(0, 1)
plt.suptitle('Filter decoded output', fontsize=16)


# Plot the spiking output of the ensemble
plt.figure(figsize=(10, 8))
plt.subplot(221)
rasterplot(sim.trange(), sim.data[spikes])
plt.ylabel("Neuron")
plt.xlim(0, 1)
plt.suptitle('Spiking output of the ensemble', fontsize=16)

# Plot the soma voltages of the neurons
#plt.subplot(222)
plt.figure(figsize=(10, 8))
plt.plot(sim.trange(), sim.data[voltage][:, 0], 'r')
plt.xlim(0, 1);
plt.suptitle('Voltages of the neurons', fontsize=16)
plt.show()
