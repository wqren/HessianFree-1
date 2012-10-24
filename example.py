from hessian_free import HessianFree
from batch import Batch
from network import DeepBeliefNetwork

import numpy

### Create the data

xor_data = [ [0.0,0.0,0.0],
			[1.0,0.0,1.0],
			[0.0,1.0,1.0],
			[1.0,1.0,0.0]]
xor_batch = Batch( [numpy.array(x).astype(numpy.float32) for x in xor_data],2,1)


##' Create the learner'

network = DeepBeliefNetwork( (2,8,9,24,1) )
learner = HessianFree(network, xor_batch, 2, 4,4, use_double=False)

### Learning stage

error = numpy.inf
iteration = 1
while error > 0.1:
	error, steps = learner.hessian_free_optimize_step()
	print error, steps
	iteration+=1

### Output

#Initial
print numpy.array([ network.predictions(x[:2]) for x in xor_data ]).flatten() 
network = learner.network
#Trained
print numpy.array([ network.predictions(x[:2]) for x in xor_data ]).flatten()